
import * as THREE from "../../libs/three.js/build/three.module.js";
import {Shaders} from "../../build/shaders/shaders.js";

export class NormalizationMaterial extends THREE.RawShaderMaterial{

	constructor(parameters = {}){
		super();

		let uniforms = {
			uDepthMap:		{ type: 't', value: null },
			uWeightMap:		{ type: 't', value: null },
		};

		this.setValues({
			uniforms: uniforms,
			vertexShader: Shaders['normalize.vs'].replace(/(#version .*)/, '$1\n' + this.getDefines()),
			fragmentShader: Shaders['normalize.fs'].replace(/(#version .*)/, '$1\n' + this.getDefines()),
		});
	}

	getDefines() {
		let defines = '';

		return defines;
	}

	updateShaderSource() {

		let vs = Shaders['normalize.vs'].replace(/(#version .*)/, '$1\n' + this.getDefines());
		let fs = Shaders['normalize.fs'].replace(/(#version .*)/, '$1\n' + this.getDefines());

		this.setValues({
			vertexShader: vs,
			fragmentShader: fs
		});

		this.needsUpdate = true;
	}

}

