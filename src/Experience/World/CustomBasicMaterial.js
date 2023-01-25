import * as THREE from 'three' 
import vertex from '../../Shaders/allBasicVertex.glsl'
import fragment from '../../Shaders/allBasicFragment.glsl'

export default class CustomBasicMaterial extends THREE.ShaderMaterial
{
    
    constructor(color)
    {
        super()
        this.lights = true,
        this.uniforms = THREE.UniformsUtils.merge([
            THREE.UniformsLib["lights"],
            THREE.UniformsLib["shadowmap"],
            THREE.UniformsUtils.clone(THREE.ShaderLib.phong.uniforms),
            {
                diffuse: { value: new THREE.Color(color) },
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
            this.uniforms.diffuse.value = new THREE.Color(color);
        }
    }

    get color() {
        return this.uniforms.diffuse.value;
    }
    
}
