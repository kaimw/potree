#version 300 es
precision mediump float;
precision mediump int;

in vec3 vColor;
in float vLinearDepth;

out vec4 fragColor;

void main() {

	//fragColor = vec4(1.0, 0.0, 0.0, 1.0);
	//fragColor = vec4(vColor, 1.0);
	//fragColor = vec4(vLinearDepth, pow(vLinearDepth, 2.0), 0.0, 1.0);
	fragColor = vec4(vLinearDepth, vLinearDepth / 30.0, vLinearDepth / 30.0, 1.0);
	
}


