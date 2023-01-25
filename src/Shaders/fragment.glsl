#include <common>
#include <lights_pars_begin>



uniform vec3 diffuseColor;
uniform float uGlossiness;
uniform vec3 rimColor;
uniform float rimPower;
uniform samplerCube envMap; 

varying vec2 vUv;
varying vec3 vNormal;
varying vec3 vViewDir;
varying vec3 vReflectVec;



void main()
{
    float NdotL = dot(vNormal, directionalLights[0].direction);
    NdotL = max(NdotL, 0.0);

     //envMap
    vec4 envColor = textureCube(envMap, vReflectVec);

    //directional
    vec3 directionalLight = directionalLights[0].color * NdotL;

  

    //ambient
    vec3 ambient = ambientLightColor;

    //specular
    vec3  H = normalize(directionalLights[0].direction + vViewDir); //halfVector
    float NdotH = dot(vNormal, H);
    NdotH = max(NdotH,0.0);

    float specularIntensity = pow(NdotH, 1000.0 / uGlossiness);

    //rim
    float rimDot = dot(vViewDir, vNormal);
    rimDot = 1.0 - rimDot;
    vec3 rim = rimColor * pow(rimDot, rimPower);

    //directionalLight = directionalLight + specularIntensity + ambient;
   // directionalLight = directionalLight + specularIntensity + 0.2 + rim;
    directionalLight = directionalLight + specularIntensity + 0.2;

    //envMap
    //vec4 envColor = textureCube(envMap, vReflectVec);

    vec3 color = diffuseColor * directionalLight;

    // vec3 finalColor = vec3(envColor) + diffuseColor * directionalLight;


    gl_FragColor = vec4(color,1.0);
}


