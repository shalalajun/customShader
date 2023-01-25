import * as THREE from 'three'
import Experience from '../Experience.js'

export default class CustomMaterial extends THREE.MeshStandardMaterial
{
    constructor()
    {
        super()
        this.onBeforeCompile = this.onBeforeCompile.bind(this);
        this.side = THREE.DoubleSide
    }

    onBeforeCompile(shader)
    {
        shader.fragmentShader = shader.fragmentShader.replace(
            "#include <output_fragment>",
            `
            #ifdef OPAQUE
            diffuseColor.a = 1.0;
            #endif
            // https://github.com/mrdoob/three.js/pull/22425
            #ifdef USE_TRANSMISSION
            diffuseColor.a *= material.transmissionAlpha + 0.1;
            #endif
            gl_FragColor = vec4( 1.0,1.0,0.0, diffuseColor.a );
        `
        )
    }
}