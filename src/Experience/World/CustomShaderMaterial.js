import * as THREE from 'three' 
import vertex from '../../Shaders/vertex.glsl'
import fragment from '../../Shaders/fragment.glsl'
import Experience from '../Experience.js'


export default class CustomShaderMaterial extends THREE.ShaderMaterial
{
    
    constructor(options)
    {
        super()

        this.loader = new THREE.TextureLoader()
        this.rampTex = this.loader.load('textures/ramp7.png')
       // this.rampTex.encoding = THREE.sRGBEncoding
        this.rampTex.flipY = false;
        console.log(this.rampTex)
        
        
        this.girlTex = this.loader.load('textures/girlTex.png')
        this.girlTex.encoding = THREE.sRGBEncoding
        this.girlTex.wrapS = THREE.RepeatWrapping;
        this.girlTex.wrapT = THREE.RepeatWrapping;
        this.girlTex.repeat.set( 1, 1 );
        this.girlTex.flipY = false;
        
        console.log(this.girlTex)


        this.lights = true,
        this.uniforms = THREE.UniformsUtils.merge([

            THREE.UniformsLib["lights"],
            THREE.UniformsLib["shadowmap"],
            
            {
                //diffuseColor: { value: new THREE.Color(options.color) },
                // envMap: { value : options.envMap },
                // uGlossiness: { value : 2.0 },
                // rimColor: { value: new THREE.Color(options.rimColor) },
                // rimPower: { value: 5.0},
                rampTex : { value: this.rampTex},
                girlTex : { value: this.girlTex}
            }
        ]);

        this.side = THREE.DoubleSide;
    
        this.vertexShader = vertex;
    
        this.fragmentShader = fragment;
    
    }
    
    updateUniforms(delta) {
        // this.uniforms.time.value += delta;
    }
    
    // set color(color) {
    //     if (this.uniforms) {
    //         this.uniforms.diffuseColor.value = new THREE.Color(color);
    //     }
    // }

    // set rimColor(color) {
    //     if (this.uniforms) {
    //         this.uniforms.rimColor.value = new THREE.Color(color);
    //     }
    // }

    // get color() {
    //     return this.uniforms.diffuseColor.value;
    // }

    // get rimColor() {
    //     return this.uniforms.rimColor.value;
    // }
}


/**
 * ????????? ?????????????????? ???????????? ???????????? ???????????? ????????????
 */






/**
 * THREE.UniformsLib["ambient"] : ambient light ??????
    THREE.UniformsLib["directional"] : directional light ??????
    THREE.UniformsLib["point"] : point light ??????
    THREE.UniformsLib["spot"] : spot light ??????
    THREE.UniformsLib["hemisphere"] : hemisphere light ??????
    THREE.UniformsLib["shadowmap"] : shadowmap ??????
    THREE.UniformsLib["fog"] : fog ??????
    THREE.UniformsLib["lights"] : ?????? ?????? ????????? ??????
 */