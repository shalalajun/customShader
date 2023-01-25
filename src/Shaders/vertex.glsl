#include <common>
#include <shadowmap_pars_vertex>

varying vec2 vUv;
varying vec3 vNormal;
varying vec3 vViewDir;
varying vec3 vReflectVec;

void main()
{
    #include <beginnormal_vertex>
    #include <defaultnormal_vertex>

    #include <begin_vertex>

    #include <worldpos_vertex>
    #include <shadowmap_vertex>

    
    vec4 modelPosition = modelMatrix * vec4(position, 1.0);
    vec4 viewPosition = viewMatrix * modelPosition;
    vec4 clipPosition = projectionMatrix * viewPosition;

    vUv = uv;
    vNormal = normalize(normalMatrix * normal);
    //vNormal =  normal;
    vViewDir = normalize(-viewPosition.xyz);

    vec3 viewPos = (viewMatrix * vec4(position, 1.0 )).xyz; // viewPosition은 각 vertex가 카메라에서 어떻게 보이는지를 나타냄
    vReflectVec = reflect(normalize(viewPos - position), normal);

    gl_Position = clipPosition;
}