import * as THREE from 'three' 
import vertex from '../../Shaders/vertex.glsl'
import fragment from '../../Shaders/fragment.glsl'

export default class CustomShaderMaterial extends THREE.ShaderMaterial
{
    
    constructor(options)
    {
        super()

        

        this.lights = true,
        this.uniforms = THREE.UniformsUtils.merge([
            THREE.UniformsLib["lights"],
            THREE.UniformsLib["shadowmap"],
            {
                diffuseColor: { value: new THREE.Color(options.color) },
                envMap: { value : options.envMap },
                uGlossiness: { value : 0.0 },
                rimColor: { value: new THREE.Color(options.rimColor) },
                rimPower: { value: 5.0}
            }
        ]);

        this.side = THREE.DoubleSide;
    
        this.vertexShader = vertex;
    
        this.fragmentShader = fragment;
    
    }
    
    updateUniforms(delta) {
        // this.uniforms.time.value += delta;
    }
    
    set color(color) {
        if (this.uniforms) {
            this.uniforms.diffuseColor.value = new THREE.Color(color);
        }
    }

    set rimColor(color) {
        if (this.uniforms) {
            this.uniforms.rimColor.value = new THREE.Color(color);
        }
    }

    get color() {
        return this.uniforms.diffuseColor.value;
    }

    get rimColor() {
        return this.uniforms.rimColor.value;
    }
}


/**
 * 쉐이더 메터리얼에서 처음부터 라이트와 쉐도우맵 구성하기
 */






/**
 * THREE.UniformsLib["ambient"] : ambient light 정보
    THREE.UniformsLib["directional"] : directional light 정보
    THREE.UniformsLib["point"] : point light 정보
    THREE.UniformsLib["spot"] : spot light 정보
    THREE.UniformsLib["hemisphere"] : hemisphere light 정보
    THREE.UniformsLib["shadowmap"] : shadowmap 정보
    THREE.UniformsLib["fog"] : fog 정보
    THREE.UniformsLib["lights"] : 위의 모든 라이트 정보
 */