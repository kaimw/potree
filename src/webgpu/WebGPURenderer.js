import {WebGPUSupport} from './WebGPUSupport.js';
import {
	pointCloudVertexWGSL, pointCloudFragmentWGSL,
	depthPassVertexWGSL, depthPassFragmentWGSL,
	attributePassVertexWGSL, attributePassFragmentWGSL,
	normalizeVertexWGSL, normalizeFragmentWGSL,
} from './PointCloudWGSL.js';
import {PointSizeType} from "../defines.js";

export class WebGPURenderer {
	constructor(viewer) {
		this.viewer = viewer;
		this.adapter = null;
		this.device = null;
		this.context = null;
		this.gpuCanvas = null;
		this.colorFormat = 'bgra8unorm';
		this.initialized = false;
		this.initPromise = null;
		this.fallback = true;

		// Standard rendering
		this.pipeline = null;
		this.uniformBuffer = null;
		this.uniformBindGroup = null;
		this.renderPassDescriptor = null;
		this.depthTexture = null;
		this.depthTextureSize = {width: 0, height: 0};
		this.pointBuffers = new Map();
		this.visibilityBuffer = null;
		this.visibilityBufferData = new Uint32Array(8192);

		// HQ rendering (WBOIT)
		this.hqDepthPipeline = null;
		this.hqAttributePipeline = null;
		this.hqNormalizePipeline = null;
		this.hqNormalizeBindGroup = null;
		this.hqDepthRT = null;       // {color, depthTexture, size}
		this.hqAttributeRT = null;   // {color, size}
		this.hqSampler = null;

		// Dynamic uniform buffer offset support
		this.uniformByteStride = 256;
		this.maxUniformNodes = 0;
	}

	async init() {
		if (this.initialized || this.initPromise) {
			return this.initPromise;
		}

		this.initPromise = this._initialize();
		return this.initPromise;
	}

	async _initialize() {
		if (!WebGPUSupport.isSupported()) {
			console.warn('WebGPU is not supported in this browser. Falling back to WebGL.');
			this.initialized = true;
			return;
		}

		const support = await WebGPUSupport.init();
		if (!support) {
			console.warn('WebGPU support probe failed. Falling back to WebGL.');
			this.initialized = true;
			return;
		}

		this.adapter = support.adapter;
		this.device = support.device;

		this.gpuCanvas = document.createElement('canvas');
		this.gpuCanvas.style.position = 'absolute';
		this.gpuCanvas.style.left = '0';
		this.gpuCanvas.style.top = '0';
		this.gpuCanvas.style.width = '100%';
		this.gpuCanvas.style.height = '100%';
		this.gpuCanvas.style.pointerEvents = 'none';
		this.gpuCanvas.style.zIndex = '2';
		this.gpuCanvas.style.backgroundColor = 'transparent';

		const renderArea = this.viewer.renderArea;
		renderArea.insertBefore(this.gpuCanvas, this.viewer.renderer.domElement);

		this.context = this.gpuCanvas.getContext('webgpu');
		this.colorFormat = (navigator.gpu && navigator.gpu.getPreferredCanvasFormat)
			? navigator.gpu.getPreferredCanvasFormat()
			: 'bgra8unorm';

		if (!this.context) {
			console.warn('Unable to create WebGPU canvas context. Falling back to WebGL.');
			this.fallback = true;
			this.initialized = true;
			return;
		}

		this.context.configure({
			device: this.device,
			format: this.colorFormat,
			alphaMode: 'premultiplied',
		});

		this.renderPassDescriptor = {
			colorAttachments: [{
				view: null,
				clearValue: {r: 0, g: 0, b: 0, a: 0},
				loadOp: 'clear',
				storeOp: 'store',
				blend: {
					color: {srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add'},
					alpha: {srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add'}
				}
			}],
			depthStencilAttachment: {
				view: null,
				depthLoadOp: 'clear',
				depthClearValue: 1.0,
				depthStoreOp: 'store'
			}
		};

		// Get the minimum uniform buffer offset alignment from the device.
		const minAlignment = this.device.limits.minUniformBufferOffsetAlignment;
		this.uniformByteStride = 128;
		while (this.uniformByteStride < minAlignment) {
			this.uniformByteStride *= 2;
		}

		// ── STANDARD RENDERING SETUP ──

		const bindGroupLayout = this.device.createBindGroupLayout({
			entries: [{
				binding: 0,
				visibility: GPUShaderStage.VERTEX,
				buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 128 }
			}, {
				binding: 1,
				visibility: GPUShaderStage.VERTEX,
				buffer: { type: 'read-only-storage', hasDynamicOffset: false }
			}]
		});

		const pipelineLayout = this.device.createPipelineLayout({
			bindGroupLayouts: [bindGroupLayout]
		});

		this.visibilityBuffer = this.device.createBuffer({
			label: 'WebGPU Visible Nodes Buffer',
			size: this.visibilityBufferData.byteLength,
			usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
		});

		this._createUniformBuffer(64);

		this.pipeline = this.device.createRenderPipeline({
			layout: pipelineLayout,
			vertex: {
				module: this.device.createShaderModule({code: pointCloudVertexWGSL}),
				entryPoint: 'main',
				buffers: [{
					arrayStride: 24,
					stepMode: 'instance',
					attributes: [
						{shaderLocation: 0, offset: 0, format: 'float32x3'},
						{shaderLocation: 1, offset: 12, format: 'float32x3'}
					]
				}]
			},
			fragment: {
				module: this.device.createShaderModule({code: pointCloudFragmentWGSL}),
				entryPoint: 'main',
				targets: [{format: this.colorFormat, blend: {
					color: {srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add'},
					alpha: {srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add'}
				}, writeMask: GPUColorWrite.ALL}]
			},
			primitive: {
				topology: 'triangle-strip',
				frontFace: 'ccw',
				cullMode: 'none'
			},
			depthStencil: {
				format: 'depth24plus',
				depthWriteEnabled: true,
				depthCompare: 'less'
			}
		});

		this.uniformBindGroup = this.device.createBindGroup({
			layout: bindGroupLayout,
			entries: [{
				binding: 0,
				resource: { buffer: this.uniformBuffer, offset: 0, size: 128 }
			}, {
				binding: 1,
				resource: { buffer: this.visibilityBuffer }
			}]
		});

		// ── HQ RENDERING SETUP (WBOIT) ──

		// Shared sampler for HQ passes
		this.hqSampler = this.device.createSampler({
			magFilter: 'nearest',
			minFilter: 'nearest',
		});

		// HQ depth pass pipeline (circular points, extended depth)
		const hqDepthPipelineLayout = this.device.createPipelineLayout({
			bindGroupLayouts: [bindGroupLayout]
		});

		this.hqDepthPipeline = this.device.createRenderPipeline({
			layout: hqDepthPipelineLayout,
			vertex: {
				module: this.device.createShaderModule({code: depthPassVertexWGSL}),
				entryPoint: 'main',
				buffers: [{
					arrayStride: 24,
					stepMode: 'instance',
					attributes: [
						{shaderLocation: 0, offset: 0, format: 'float32x3'},
						{shaderLocation: 1, offset: 12, format: 'float32x3'}
					]
				}]
			},
			fragment: {
				module: this.device.createShaderModule({code: depthPassFragmentWGSL}),
				entryPoint: 'main',
				targets: [{format: 'rgba16float', writeMask: GPUColorWrite.ALL}]
			},
			primitive: {
				topology: 'triangle-strip',
				frontFace: 'ccw',
				cullMode: 'none'
			},
			depthStencil: {
				format: 'depth24plus',
				depthWriteEnabled: true,
				depthCompare: 'less'
			}
		});

		// HQ attribute pass pipeline (circular points, weighted additive blend)
		const hqAttributePipelineLayout = this.device.createPipelineLayout({
			bindGroupLayouts: [bindGroupLayout]
		});

		this.hqAttributePipeline = this.device.createRenderPipeline({
			layout: hqAttributePipelineLayout,
			vertex: {
				module: this.device.createShaderModule({code: attributePassVertexWGSL}),
				entryPoint: 'main',
				buffers: [{
					arrayStride: 24,
					stepMode: 'instance',
					attributes: [
						{shaderLocation: 0, offset: 0, format: 'float32x3'},
						{shaderLocation: 1, offset: 12, format: 'float32x3'}
					]
				}]
			},
			fragment: {
				module: this.device.createShaderModule({code: attributePassFragmentWGSL}),
				entryPoint: 'main',
				targets: [{format: 'rgba16float', blend: {
					color: {srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add'},
					alpha: {srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add'}
				}, writeMask: GPUColorWrite.ALL}]
			},
			primitive: {
				topology: 'triangle-strip',
				frontFace: 'ccw',
				cullMode: 'none'
			},
			depthStencil: {
				format: 'depth24plus',
				depthWriteEnabled: false,
				depthCompare: 'less'
			}
		});

		// HQ normalization pipeline (fullscreen combine)
		const normalizeBindGroupLayout = this.device.createBindGroupLayout({
			entries: [{
				binding: 0,
				visibility: GPUShaderStage.FRAGMENT,
				texture: { sampleType: 'unfilterable-float' }
			}, {
				binding: 1,
				visibility: GPUShaderStage.FRAGMENT,
				texture: { sampleType: 'unfilterable-float' }
			}, {
				binding: 2,
				visibility: GPUShaderStage.FRAGMENT,
				sampler: { type: 'non-filtering' }
			}]
		});

		const normalizePipelineLayout = this.device.createPipelineLayout({
			bindGroupLayouts: [normalizeBindGroupLayout]
		});

		this.hqNormalizePipeline = this.device.createRenderPipeline({
			layout: normalizePipelineLayout,
			vertex: {
				module: this.device.createShaderModule({code: normalizeVertexWGSL}),
				entryPoint: 'main',
			},
			fragment: {
				module: this.device.createShaderModule({code: normalizeFragmentWGSL}),
				entryPoint: 'main',
				targets: [{format: this.colorFormat, writeMask: GPUColorWrite.ALL}]
			},
			primitive: {
				topology: 'triangle-list',
				frontFace: 'ccw',
				cullMode: 'none'
			},
			depthStencil: {
				format: 'depth24plus',
				depthWriteEnabled: true,
				depthCompare: 'always'
			}
		});

		this.fallback = false;
		this.initialized = true;
	}

	_createUniformBuffer(capacity) {
		const newSize = capacity * this.uniformByteStride;
		this.maxUniformNodes = capacity;

		const newBuffer = this.device.createBuffer({
			label: 'WebGPU CameraUniformBuffer',
			size: newSize,
			usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
		});

		if (this.uniformBuffer) {
			this.uniformBuffer.destroy();
		}
		this.uniformBuffer = newBuffer;
	}

	_ensureUniformBufferCapacity(nodeCount) {
		if (nodeCount <= this.maxUniformNodes) {
			return false;
		}

		const newCapacity = Math.max(nodeCount, this.maxUniformNodes * 2);
		this._createUniformBuffer(newCapacity);
		this._recreateBindGroup();
		return true;
	}

	_recreateBindGroup() {
		if (!this.pipeline) return;

		const bindGroupLayout = this.pipeline.getBindGroupLayout(0);
		this.uniformBindGroup = this.device.createBindGroup({
			layout: bindGroupLayout,
			entries: [{
				binding: 0,
				resource: { buffer: this.uniformBuffer, offset: 0, size: 128 }
			}, {
				binding: 1,
				resource: { buffer: this.visibilityBuffer }
			}]
		});
	}

	_resizeHQTargets(width, height) {
		const w = Math.max(1, width);
		const h = Math.max(1, height);

		if (this.hqDepthRT && this.hqDepthRT.size.width === w && this.hqDepthRT.size.height === h) {
			return; // already correct size
		}

		// Destroy old textures if they exist
		if (this.hqDepthRT) {
			this.hqDepthRT.color.destroy();
			this.hqDepthRT.depthTexture.destroy();
		}
		if (this.hqAttributeRT) {
			this.hqAttributeRT.color.destroy();
		}

		// Depth RT: color (for EDL depth map) + depth texture
		const depthColorTexture = this.device.createTexture({
			size: {width: w, height: h, depthOrArrayLayers: 1},
			format: 'rgba16float',
			usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
		});

		const depthDepthTexture = this.device.createTexture({
			size: {width: w, height: h, depthOrArrayLayers: 1},
			format: 'depth24plus',
			usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
		});

		this.hqDepthRT = {
			color: depthColorTexture,
			depthTexture: depthDepthTexture,
			size: {width: w, height: h}
		};

		// Attribute RT: rgba16float required for WBOIT accumulation (values can exceed 1.0)
		const attributeColorTexture = this.device.createTexture({
			size: {width: w, height: h, depthOrArrayLayers: 1},
			format: 'rgba16float',
			usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
		});

		this.hqAttributeRT = {
			color: attributeColorTexture,
			size: {width: w, height: h}
		};

		// Recreate normalization bind group with new textures
		this.hqNormalizeBindGroup = this.device.createBindGroup({
			layout: this.hqNormalizePipeline.getBindGroupLayout(0),
			entries: [{
				binding: 0,
				resource: this.hqAttributeRT.color.createView()
			}, {
				binding: 1,
				resource: this.hqDepthRT.color.createView()
			}, {
				binding: 2,
				resource: this.hqSampler
			}]
		});
	}

	ensureSize() {
		const width = Math.max(1, this.viewer.renderArea.clientWidth);
		const height = Math.max(1, this.viewer.renderArea.clientHeight);

		if (this.gpuCanvas.width !== width || this.gpuCanvas.height !== height) {
			this.gpuCanvas.width = width;
			this.gpuCanvas.height = height;
		}

		if (this.depthTextureSize.width !== width || this.depthTextureSize.height !== height) {
			this.depthTextureSize.width = width;
			this.depthTextureSize.height = height;
			this.depthTexture = this.device.createTexture({
				size: {width, height, depthOrArrayLayers: 1},
				format: 'depth24plus',
				usage: GPUTextureUsage.RENDER_ATTACHMENT
			});
		}

		this.renderPassDescriptor.colorAttachments[0].view = this.context.getCurrentTexture().createView();
		this.renderPassDescriptor.depthStencilAttachment.view = this.depthTexture.createView();

		return {width, height};
	}

	createPointBuffers(node) {
		const geometry = node.geometryNode && node.geometryNode.geometry;
		if (!geometry || !geometry.attributes || !geometry.attributes.position) {
			return null;
		}

		const key = geometry.id;
		if (this.pointBuffers.has(key)) {
			return this.pointBuffers.get(key);
		}

		const positionAttr = geometry.attributes.position;
		const pointCount = positionAttr.count;

		let colorAttr = geometry.attributes.color || geometry.attributes.rgb || geometry.attributes.rgba;
		const vertexData = new Float32Array(pointCount * 6);

		for (let i = 0; i < pointCount; i++) {
			vertexData[i * 6 + 0] = positionAttr.array[i * positionAttr.itemSize + 0];
			vertexData[i * 6 + 1] = positionAttr.array[i * positionAttr.itemSize + 1];
			vertexData[i * 6 + 2] = positionAttr.array[i * positionAttr.itemSize + 2];

			if (colorAttr) {
				if (colorAttr.itemSize >= 3) {
					vertexData[i * 6 + 3] = colorAttr.array[i * colorAttr.itemSize + 0] / (colorAttr.array instanceof Uint8Array ? 255.0 : 1.0);
					vertexData[i * 6 + 4] = colorAttr.array[i * colorAttr.itemSize + 1] / (colorAttr.array instanceof Uint8Array ? 255.0 : 1.0);
					vertexData[i * 6 + 5] = colorAttr.array[i * colorAttr.itemSize + 2] / (colorAttr.array instanceof Uint8Array ? 255.0 : 1.0);
				} else {
					vertexData[i * 6 + 3] = 1.0;
					vertexData[i * 6 + 4] = 1.0;
					vertexData[i * 6 + 5] = 1.0;
				}
			} else {
				vertexData[i * 6 + 3] = 1.0;
				vertexData[i * 6 + 4] = 1.0;
				vertexData[i * 6 + 5] = 1.0;
			}
		}

		const gpuBuffer = this.device.createBuffer({
			label: 'WebGPU PointCloud Buffer',
			size: vertexData.byteLength,
			usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
			mappedAtCreation: true
		});

		new Float32Array(gpuBuffer.getMappedRange()).set(vertexData);
		gpuBuffer.unmap();

		const pointState = {vertexBuffer: gpuBuffer, count: pointCount};
		this.pointBuffers.set(key, pointState);
		return pointState;
	}

	_fillNodeUniforms(outArray, slotIndex, mvpMatrix, width, height, pointSize, nodeSpacing, fov, minSize, maxSize, pointSizeType, nodeLevel, vnStart, octreeSize, useVisibilityBuffer) {
		const elements = mvpMatrix.elements;
		const strideFloats = this.uniformByteStride / 4;
		const offset = slotIndex * strideFloats;

		outArray[offset + 0] = elements[0];
		outArray[offset + 1] = elements[1];
		outArray[offset + 2] = elements[2];
		outArray[offset + 3] = elements[3];
		outArray[offset + 4] = elements[4];
		outArray[offset + 5] = elements[5];
		outArray[offset + 6] = elements[6];
		outArray[offset + 7] = elements[7];
		outArray[offset + 8] = elements[8];
		outArray[offset + 9] = elements[9];
		outArray[offset + 10] = elements[10];
		outArray[offset + 11] = elements[11];
		outArray[offset + 12] = elements[12];
		outArray[offset + 13] = elements[13];
		outArray[offset + 14] = elements[14];
		outArray[offset + 15] = elements[15];
		outArray[offset + 16] = width;
		outArray[offset + 17] = height;
		outArray[offset + 18] = pointSize;
		outArray[offset + 19] = nodeSpacing;
		outArray[offset + 20] = fov;
		outArray[offset + 21] = minSize;
		outArray[offset + 22] = maxSize;
		outArray[offset + 23] = pointSizeType;
		outArray[offset + 24] = nodeLevel;
		outArray[offset + 25] = vnStart;
		outArray[offset + 26] = octreeSize;
		outArray[offset + 27] = useVisibilityBuffer ? 1.0 : 0.0;
	}

	updateVisibilityBuffer(pointcloud, camera) {
		if (!this.visibilityBuffer || !pointcloud.computeVisibilityTextureData) {
			return null;
		}

		const visibilityData = pointcloud.computeVisibilityTextureData(pointcloud.visibleNodes, camera);
		if (visibilityData.data.length > this.visibilityBufferData.length * 4) {
			console.warn('Visibility data exceeds buffer size, LOD may be incorrect');
			return null;
		}

		this.visibilityBufferData.fill(0);
		for (let i = 0; i < visibilityData.data.length; i += 4) {
			const nodeIndex = i / 4;
			this.visibilityBufferData[nodeIndex] =
				visibilityData.data[i + 0] |
				(visibilityData.data[i + 1] << 8) |
				(visibilityData.data[i + 2] << 16) |
				(visibilityData.data[i + 3] << 24);
		}

		this.device.queue.writeBuffer(this.visibilityBuffer, 0, this.visibilityBufferData.buffer, 0, this.visibilityBufferData.byteLength);

		return visibilityData;
	}

	clear() {
		const {renderer} = this.viewer;

		if (this.viewer.background === "skybox") {
			renderer.setClearColor(0xff0000, 1);
		} else if (this.viewer.background === "gradient") {
			renderer.setClearColor(0x00ff00, 1);
		} else if (this.viewer.background === "black") {
			renderer.setClearColor(0x000000, 1);
		} else if (this.viewer.background === "white") {
			renderer.setClearColor(0xFFFFFF, 1);
		} else {
			renderer.setClearColor(0x000000, 0);
		}

		renderer.clear();

		if (!this.initialized || this.fallback || !this.device) {
			return;
		}

		const size = this.ensureSize();
		const commandEncoder = this.device.createCommandEncoder();
		const passEncoder = commandEncoder.beginRenderPass(this.renderPassDescriptor);
		passEncoder.end();
		this.device.queue.submit([commandEncoder.finish()]);
	}

	render(scene, camera, target = null, params = {}) {
		if (!this.initialized) {
			this.init().catch(() => {});
		}

		if (this.fallback || !this.device) {
			let actualScene = scene;
			let actualCamera = camera;
			let actualParams = params;

			if (scene && typeof scene.render === 'function' && !scene.isScene) {
				actualScene = this.viewer.scene.scenePointCloud;
				actualCamera = this.viewer.scene.getActiveCamera();
				actualParams = {};
			}

			return this.viewer.pRenderer.render(actualScene, actualCamera, target, actualParams);
		}

		this.renderThreePass(camera);

		if (this.viewer.useHQ) {
			this.renderPointCloudsHQ(camera);
		} else {
			this.renderPointClouds(camera);
		}
	}

	renderThreePass(camera) {
		const {viewer} = this;
		const renderer = viewer.renderer;
		const activeCamera = camera && camera.isCamera ? camera : viewer.scene.getActiveCamera();

		viewer.dispatchEvent({type: "render.pass.begin", viewer: viewer});

		if (viewer.background === "skybox") {
			viewer.skybox.camera.rotation.copy(viewer.scene.cameraP.rotation);
			viewer.skybox.camera.fov = viewer.scene.cameraP.fov;
			viewer.skybox.camera.aspect = viewer.scene.cameraP.aspect;

			viewer.skybox.parent.rotation.x = 0;
			viewer.skybox.parent.updateMatrixWorld();

			viewer.skybox.camera.updateProjectionMatrix();
			renderer.render(viewer.skybox.scene, viewer.skybox.camera);
		} else if (viewer.background === "gradient") {
			renderer.render(viewer.scene.sceneBG, viewer.scene.cameraBG);
		}

		for (let pointcloud of viewer.scene.pointclouds) {
			pointcloud.material.useEDL = false;
		}

		renderer.render(viewer.scene.scene, activeCamera);

		viewer.dispatchEvent({type: "render.pass.scene", viewer: viewer});

		viewer.clippingTool.update();
		renderer.render(viewer.clippingTool.sceneMarker, viewer.scene.cameraScreenSpace);
		renderer.render(viewer.clippingTool.sceneVolume, activeCamera);

		renderer.render(viewer.controls.sceneControls, activeCamera);
		renderer.clearDepth();

		viewer.transformationTool.update();

		viewer.dispatchEvent({type: "render.pass.perspective_overlay", viewer: viewer});
		viewer.dispatchEvent({type: "render.pass.end", viewer: viewer});
	}

	// ── Collect drawable nodes and fill uniform buffer ──

	_collectDrawNodes(pointclouds, camera, outDrawNodes) {
		const rendererCamera = camera && camera.isCamera ? camera : this.viewer.scene.getActiveCamera();
		const size = this.ensureSize();
		const strideFloats = this.uniformByteStride / 4;
		let totalNodeCount = 0;

		for (const pointcloud of pointclouds) {
			for (const node of pointcloud.visibleNodes) {
				if (node.geometryNode && node.geometryNode.geometry && node.geometryNode.geometry.attributes.position) {
					totalNodeCount++;
				}
			}
		}

		if (totalNodeCount === 0) {
			return 0;
		}

		this._ensureUniformBufferCapacity(totalNodeCount);

		const allUniformData = new Float32Array(this.maxUniformNodes * strideFloats);
		allUniformData.fill(0);

		let nodeIndex = 0;

		for (const pointcloud of pointclouds) {
			const material = pointcloud.material;
			const useVisibilityBuffer = material.pointSizeType === PointSizeType.ADAPTIVE;

			const visibilityData = useVisibilityBuffer
				? this.updateVisibilityBuffer(pointcloud, rendererCamera)
				: null;

			const boundingBox = pointcloud.pcoGeometry && pointcloud.pcoGeometry.boundingBox;
			const octreeSize = boundingBox ? boundingBox.max.x - boundingBox.min.x : 1.0;

			for (const node of pointcloud.visibleNodes) {
				if (!node.geometryNode || !node.geometryNode.geometry || !node.geometryNode.geometry.attributes.position) {
					continue;
				}

				const pointState = this.createPointBuffers(node);
				if (!pointState || pointState.count === 0) {
					continue;
				}

				const modelMatrix = node.sceneNode.matrixWorld;
				const vp = rendererCamera.projectionMatrix.clone().multiply(rendererCamera.matrixWorldInverse);
				const mvpMatrix = vp.clone().multiply(modelMatrix);
				const nodeSpacing = node.geometryNode.estimatedSpacing || pointcloud.pcoGeometry.spacing || 1.0;
				const fov = rendererCamera.fov ? Math.PI * rendererCamera.fov / 180 : 1.0;
				const nodeLevel = node.getLevel ? node.getLevel() : 0;
				const vnStart = visibilityData ? visibilityData.offsets.get(node) || 0 : 0;

				this._fillNodeUniforms(
					allUniformData,
					nodeIndex,
					mvpMatrix,
					size.width,
					size.height,
					material.size || 1.0,
					nodeSpacing,
					fov,
					material.minSize || 1.0,
					material.maxSize || 50.0,
					material.pointSizeType,
					nodeLevel,
					vnStart,
					octreeSize,
					!!visibilityData
				);

				outDrawNodes.push({pointState, nodeIndex, pointcloud});
				nodeIndex++;
			}
		}

		// Write all per-node uniforms to GPU
		this.device.queue.writeBuffer(
			this.uniformBuffer,
			0,
			allUniformData.buffer,
			0,
			allUniformData.byteLength
		);

		return totalNodeCount;
	}

	// ── Standard single-pass rendering ──

	renderPointClouds(camera) {
		const pointclouds = this.viewer.scene.pointclouds.filter(pc => pc.visible);
		if (pointclouds.length === 0) {
			return;
		}

		const drawNodes = [];
		const totalNodeCount = this._collectDrawNodes(pointclouds, camera, drawNodes);
		if (totalNodeCount === 0) {
			return;
		}

		const commandEncoder = this.device.createCommandEncoder();
		const passEncoder = commandEncoder.beginRenderPass(this.renderPassDescriptor);
		passEncoder.setPipeline(this.pipeline);

		for (const dn of drawNodes) {
			const dynamicOffset = dn.nodeIndex * this.uniformByteStride;
			passEncoder.setBindGroup(0, this.uniformBindGroup, [dynamicOffset]);
			passEncoder.setVertexBuffer(0, dn.pointState.vertexBuffer);
			passEncoder.draw(4, dn.pointState.count, 0, 0);
		}

		passEncoder.end();
		this.device.queue.submit([commandEncoder.finish()]);
	}

	// ── HQ multi-pass WBOIT rendering ──

	renderPointCloudsHQ(camera) {
		const pointclouds = this.viewer.scene.pointclouds.filter(pc => pc.visible);
		if (pointclouds.length === 0) {
			return;
		}

		const size = this.ensureSize();
		this._resizeHQTargets(size.width, size.height);

		const drawNodes = [];
		const totalNodeCount = this._collectDrawNodes(pointclouds, camera, drawNodes);
		if (totalNodeCount === 0) {
			return;
		}

		// ── Pass 1: Depth Pass ──
		{
			const commandEncoder = this.device.createCommandEncoder();

			const depthPassDescriptor = {
				colorAttachments: [{
					view: this.hqDepthRT.color.createView(),
					clearValue: {r: 0, g: 0, b: 0, a: 0},
					loadOp: 'clear',
					storeOp: 'store',
				}],
				depthStencilAttachment: {
					view: this.hqDepthRT.depthTexture.createView(),
					depthLoadOp: 'clear',
					depthClearValue: 1.0,
					depthStoreOp: 'store',
				}
			};

			const passEncoder = commandEncoder.beginRenderPass(depthPassDescriptor);
			passEncoder.setPipeline(this.hqDepthPipeline);

			for (const dn of drawNodes) {
				const dynamicOffset = dn.nodeIndex * this.uniformByteStride;
				passEncoder.setBindGroup(0, this.uniformBindGroup, [dynamicOffset]);
				passEncoder.setVertexBuffer(0, dn.pointState.vertexBuffer);
				passEncoder.draw(4, dn.pointState.count, 0, 0);
			}

			passEncoder.end();
			this.device.queue.submit([commandEncoder.finish()]);
		}

		// ── Pass 2: Attribute Pass (weighted color accumulation) ──
		{
			const commandEncoder = this.device.createCommandEncoder();

			const attrPassDescriptor = {
				colorAttachments: [{
					view: this.hqAttributeRT.color.createView(),
					clearValue: {r: 0, g: 0, b: 0, a: 0},
					loadOp: 'clear',
					storeOp: 'store',
				}],
				depthStencilAttachment: {
					view: this.hqDepthRT.depthTexture.createView(),
					depthLoadOp: 'load',
					depthClearValue: 1.0,
					depthStoreOp: 'store',
				}
			};

			const passEncoder = commandEncoder.beginRenderPass(attrPassDescriptor);
			passEncoder.setPipeline(this.hqAttributePipeline);

			for (const dn of drawNodes) {
				const dynamicOffset = dn.nodeIndex * this.uniformByteStride;
				passEncoder.setBindGroup(0, this.uniformBindGroup, [dynamicOffset]);
				passEncoder.setVertexBuffer(0, dn.pointState.vertexBuffer);
				passEncoder.draw(4, dn.pointState.count, 0, 0);
			}

			passEncoder.end();
			this.device.queue.submit([commandEncoder.finish()]);
		}

		// ── Pass 3: Normalization Pass (combine onto canvas) ──
		{
			const commandEncoder = this.device.createCommandEncoder();

			const normPassDescriptor = {
				colorAttachments: [{
					view: this.context.getCurrentTexture().createView(),
					clearValue: {r: 0, g: 0, b: 0, a: 0},
					loadOp: 'clear',
					storeOp: 'store',
				}],
				depthStencilAttachment: {
					view: this.depthTexture.createView(),
					depthLoadOp: 'clear',
					depthClearValue: 1.0,
					depthStoreOp: 'store',
				}
			};

			const passEncoder = commandEncoder.beginRenderPass(normPassDescriptor);
			passEncoder.setPipeline(this.hqNormalizePipeline);
			passEncoder.setBindGroup(0, this.hqNormalizeBindGroup);
			passEncoder.draw(3, 1, 0, 0);
			passEncoder.end();
			this.device.queue.submit([commandEncoder.finish()]);
		}
	}
}
