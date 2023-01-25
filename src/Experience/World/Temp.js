import * as THREE from 'three'
import Experience from '../Experience.js'
import vertex from '../../Shaders/vertex.glsl'
import fragment from '../../Shaders/fragment.glsl'
import { MeshStandardMaterial, Side, sRGBEncoding } from 'three'
import CustomMaterial from './CustomMaterial.js'
import CustomShaderMaterial from './CustomShaderMaterial.js'
import CustomBasicMaterial from './CustomBasicMaterial.js'



export default class Temp
{
    constructor()
    {
        this.experience = new Experience()
        this.scene = this.experience.scene
        this.resources = this.experience.resources

        this.resource = this.resources.items.girlModel

        this.foxMaterial = new CustomShaderMaterial()

        this.environmentMap = {}
        this.environmentMap.texture = this.resources.items.environmentMapTexture
        this.environmentMap.texture.encoding = THREE.sRGBEncoding

        this.setTemp()
        this.setFloor()
    }

    setTemp()
    {

        // this.rampTex = this.resources.items.rampTex
        // console.log(this.rampTex)
        // this.rampTex.encoding = THREE.sRGBEncoding
        // this.rampTex.flipY = false;

        this.model = this.resource.scene
        console.log(this.model)
        this.model.scale.set(0.1,0.1,0.1)
        this.model.position.y = -0.5
       
       

        this.model.traverse((child)=>
        {
            if(child instanceof THREE.Mesh)
            {
                child.material = this.foxMaterial
                child.castShadow = true
                child.receiveShadow = true
            } 
        })

        this.scene.add(this.model)


        this.geo = new THREE.SphereGeometry(0.5,64,64)
        this.material = new CustomShaderMaterial({color:"#ff0000",envMap: this.environmentMap.texture,rimColor:"#ffffff"})
        this.material.color = "#ffff00"
        this.mesh = new THREE.Mesh(this.geo,this.material)

        //this.mesh.rotation.x = Math.PI / 2
        this.scene.add(this.mesh)
        this.mesh.castShadow = true
        this.mesh.receiveShadow = true



        this.geo = new THREE.SphereGeometry(0.5,64,64)
        this.material2 = new THREE.MeshPhongMaterial({color:'#ff0000'})
        this.mesh2 = new THREE.Mesh(this.geo,this.material2)
        this.mesh2.position.set(1.02,0,0)
        this.mesh2.castShadow = true
        this.mesh2.receiveShadow = true
        //this.mesh.rotation.x = Math.PI / 2
        this.scene.add(this.mesh2)


        this.geo = new THREE.SphereGeometry(0.5,64,64)
        this.material3 = new CustomBasicMaterial("#ff0000")
        //this.material3.color = "#ffff00"
        this.mesh3 = new THREE.Mesh(this.geo,this.material3)
        this.mesh3.position.set(-1,0,0)
        //this.mesh.rotation.x = Math.PI / 2
        this.scene.add(this.mesh3)
        this.mesh3.castShadow = true
        this.mesh3.receiveShadow = true
    }

    setFloor()
    {
        this.planGeo = new THREE.PlaneGeometry(2,2)
        this.planeMaterial = new MeshStandardMaterial({color:0xffffff})
        this.planeMesh = new THREE.Mesh(this.planGeo,this.planeMaterial)
        this.planeMesh.position.y = -0.5
        this.planeMesh.rotation.x = - Math.PI / 2
        this.scene.add(this.planeMesh)
        this.planeMesh.castShadow = true
        this.planeMesh.receiveShadow = true
    }
}