import Vue from 'vue'
import App from './App.vue'
import router from './router'
import store from './store'
import './plugins/element.js'
import axios from 'axios'
import vConsole from 'vconsole'
let Console = new vConsole()
Vue.use(Console)
Vue.config.productionTip = false
Vue.prototype.$axios = axios

new Vue({
    router,
    store,
    render: h => h(App)
}).$mount('#app')