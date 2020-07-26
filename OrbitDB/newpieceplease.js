class NewPiecePlease {
    constructor (IPFS, OrbitDB) {
      this.OrbitDB = OrbitDB
  
      this.node = new IPFS({
        preload: { enabled: false },
        repo: './ipfs',
        EXPERIMENTAL: { pubsub: true },
        config: {
          Bootstrap: [],
          Addresses: { Swarm: [] }
        }
      })
  
      this.node.on('error', (e) => { throw (e) })
      this.node.on('ready', this._init.bind(this))
    }
  
    async _init () {
      this.orbitdb = await this.OrbitDB.createInstance(this.node)
      this.onready()
    }
}

try {
    const Ipfs = require('ipfs')
    const OrbitDB = require('orbit-db')
    module.exports = exports = new NewPiecePlease(Ipfs, OrbitDB)
} catch (e) {
    window.NPP = new NewPiecePlease(window.Ipfs, window.OrbitDB)
}

