export class WebGPUSupport {
	static isSupported() {
		return typeof navigator !== 'undefined' && typeof navigator.gpu !== 'undefined' && typeof navigator.gpu.requestAdapter === 'function';
	}

	static async requestAdapter(options = {}) {
		if (!WebGPUSupport.isSupported()) {
			return null;
		}

		try {
			return await navigator.gpu.requestAdapter(options);
		} catch (e) {
			console.warn('WebGPUSupport: requestAdapter failed', e);
			return null;
		}
	}

	static async requestDevice(adapter, descriptor = {}) {
		if (!adapter) {
			return null;
		}

		try {
			return await adapter.requestDevice(descriptor);
		} catch (e) {
			console.warn('WebGPUSupport: requestDevice failed', e);
			return null;
		}
	}

	static async init(options = {}) {
		const adapter = await WebGPUSupport.requestAdapter({powerPreference: options.powerPreference || 'high-performance'});
		if (!adapter) {
			return null;
		}

		const device = await WebGPUSupport.requestDevice(adapter, options.deviceDescriptor || {});
		if (!device) {
			return null;
		}

		return {adapter, device};
	}
}
