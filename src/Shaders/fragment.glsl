#include <common>
#include <lights_pars_begin>
#include <packing>
#include <shadowmap_pars_fragment>
#include <shadowmask_pars_fragment>


uniform vec3 diffuseColor;
uniform float uGlossiness;
uniform vec3 rimColor;
uniform float rimPower;
uniform samplerCube envMap;
uniform sampler2D rampTex;  

varying vec2 vUv;
varying vec3 vNormal;
varying vec3 vViewDir;
varying vec3 vReflectVec;


vec3 BRDF_Lambert( const in vec3 diffuseColor ) {
	return RECIPROCAL_PI * diffuseColor;
} // validated

vec3 F_Schlick( const in vec3 f0, const in float f90, const in float dotVH ) {
	// Original approximation by Christophe Schlick '94
	// float fresnel = pow( 1.0 - dotVH, 5.0 );
	// Optimized variant (presented by Epic at SIGGRAPH '13)
	// https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
} // validated


float F_Schlick( const in float f0, const in float f90, const in float dotVH ) {
	// Original approximation by Christophe Schlick '94
	// float fresnel = pow( 1.0 - dotVH, 5.0 );
	// Optimized variant (presented by Epic at SIGGRAPH '13)
	// https://cdn2.unrealengine.com/Resources/files/2013SiggraphPresentationsNotes-26915738.pdf
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
} // validated


// End Rect Area Light
float G_BlinnPhong_Implicit( /* const in float dotNL, const in float dotNV */ ) {
	// geometry term is (n dot l)(n dot v) / 4(n dot l)(n dot v)
	return 0.25;
}


float D_BlinnPhong( const in float shininess, const in float dotNH ) {
	return RECIPROCAL_PI * ( shininess * 0.5 + 1.0 ) * pow( dotNH, shininess );
}


vec3 BRDF_BlinnPhong( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in vec3 specularColor, const in float shininess ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( specularColor, 1.0, dotVH );
	float G = G_BlinnPhong_Implicit( /* dotNL, dotNV */ );
	float D = D_BlinnPhong( shininess, dotNH );
	return F * ( G * D );
} // validated



vec3 fresnel(in vec3 f0, in float product)
{
    //// 0(max fres) ~ 1(min fres)
    return mix(f0, vec3(1.0), pow(1.0 - product, 5.0));
}

void main()
{

    DirectionalLightShadow directionalShadow = directionalLightShadows[0];

    float shadow = getShadow(
        directionalShadowMap[0],
        directionalShadow.shadowMapSize,
        directionalShadow.shadowBias,
        directionalShadow.shadowRadius,
        vDirectionalShadowCoord[0]
    );


    float NdotL = dot(vNormal, directionalLights[0].direction);
    NdotL = (NdotL * shadow) * 0.5 + 0.5;
   
   //vec3 NdotLCol = mix(vec3(0.0,0.0,1.0),vec3(1.0,1.0,1.0),NdotL);
  
    //directional
    vec3 directionalLight = directionalLights[0].color * NdotL;


    //ambient
    vec3 ambient = ambientLightColor;

    //specular
    vec3  H = normalize(directionalLights[0].direction + vViewDir); //halfVector
    float NdotH = dot(vNormal, H);
    NdotH = max(NdotH,0.0);
    

    float CdH = dot(vViewDir, H);
    vec3 fres = fresnel(vec3(0.5),CdH);


    // vec3 specularIntensity;
    // if(NdotL > 0.0)
    // {
    //  specularIntensity = pow(NdotH, 100.0) * vec3(0.0,1.0,1.0);
    // }
  
    vec3 specularIntensity;
    if(NdotL > 0.0)
    {
     specularIntensity = BRDF_Lambert(diffuseColor) * BRDF_BlinnPhong(directionalLights[0].direction , vViewDir , vNormal, diffuseColor, 20.0 ) * 2.0;
    }


    //rim
    float rimDot = dot(vViewDir, vNormal);
    rimDot = 1.0 - rimDot;
    vec3 rim = rimColor * pow(rimDot, rimPower);

    //directionalLight = directionalLight + specularIntensity + ambient;
   // directionalLight = directionalLight + specularIntensity + 0.2 + rim;
    directionalLight = directionalLight + specularIntensity + 0.2 + rim;

    //envMap
    //vec4 envColor = textureCube(envMap, vReflectVec);
    
    vec3 color = BRDF_Lambert(diffuseColor) * directionalLight;

    // vec3 finalColor = vec3(envColor) + diffuseColor * directionalLight;
    vec3 rimCol = texture2D(rampTex,vec2(NdotL,0.5)).xyz;


    gl_FragColor = vec4(rimCol,1.0);
}


