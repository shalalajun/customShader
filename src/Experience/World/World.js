import Experience from '../Experience.js'
import Environment from './Environment.js'
import Temp from './Temp.js'


export default class World
{
    constructor()
    {
        this.experience = new Experience()
        this.scene = this.experience.scene
        this.resources = this.experience.resources

        // Wait for resources
        this.resources.on('ready', () =>
        {
            this.temp = new Temp()
            // Setup
            this.environment = new Environment()
        })
    }

    update()
    {
        
    }
}