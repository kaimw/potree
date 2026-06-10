#version 300 es
precision highp float;
precision highp int;

in vec3	vColor;

out vec4 fragColor;

void main() {

	vec3 color = vColor;

	fragColor = vec4(color, 1.0);
}


