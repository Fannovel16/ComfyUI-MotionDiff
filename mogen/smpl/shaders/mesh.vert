#version 330 core

// Vertex Attributes
layout(location = 0) in vec3 position;
layout(location = NORMAL_LOC) in vec3 normal;
layout(location = INST_M_LOC) in mat4 inst_m;

// Uniforms
uniform mat4 M; // Model matrix
uniform mat4 V; // View matrix
uniform mat4 P; // Projection matrix

// Outputs
out vec3 frag_position;
out vec3 frag_normal;

void main()
{
    mat4 modelView = V * M * inst_m; // Compute model-view matrix
    gl_Position = P * modelView * vec4(position, 1);
    frag_position = vec3(modelView * vec4(position, 1.0)); // Position in camera space

    mat3 normalMatrix = transpose(inverse(mat3(modelView))); // Normal matrix in camera space
    frag_normal = normalize(normalMatrix * normal); // Transform normal to camera space
}