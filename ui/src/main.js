import { createApp }    from 'vue'
import { createRouter, createWebHashHistory } from 'vue-router'
import App      from './App.vue'
import Annotate from './components/Annotate.vue'
import Inspect  from './components/Inspect.vue'

const router = createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: '/',         redirect: '/annotate' },
    { path: '/annotate', component: Annotate },
    { path: '/inspect',  component: Inspect  },
  ],
})

createApp(App).use(router).mount('#app')
