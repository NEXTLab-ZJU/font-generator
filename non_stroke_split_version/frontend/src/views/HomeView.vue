<template>
    <div class="home">
        <el-steps :active="currentStep - 1" finish-status="success" style="width:40%;margin-left:30%;">
            <el-step title="Zi2Zi网络训练"></el-step>
            <el-step title="粗糙位图生成"></el-step>
            <el-step title="SD微调网络训练"></el-step>
            <el-step title="高质量位图生成"></el-step>
            <el-step title="通用矢量化"></el-step>
        </el-steps>
        
        <div style="margin: 0 auto; text-align: center;margin-top:25px;">
            <el-button @click="refresh" style="margin-bottom: 20px;"> 手动刷新进度（默认每15s自动刷新）</el-button>
            <div v-if="isHandling">
                <div style="text-align: center;margin-bottom:25px;">当前步骤进度</div>
                <div style="text-align: center;margin-bottom:25px;">
                    <el-progress type="circle" :percentage="progress * 100"></el-progress>
                </div>
            </div>
            <div v-if="!isHandling && currentStep == 0">
                <div>
                    <el-upload class="upload-demo" ref="upload" drag :action="config.request_url + '/upload'" accept="zip"
                         :on-success="uploadSuccess"
                         :auto-upload="false">
                        <el-button style="margin-top: 20%;" slot="trigger" size="small" type="primary">选取文件</el-button>
                    </el-upload>
                </div>
                <div>
                    <el-button style="margin-left: 10px;" size="small" type="success"
                        @click="submitUpload">上传到服务器</el-button>
                </div>
                <div slot="tip" class="el-upload__tip">只能上传zip文件</div>
            </div>

            <div>
                <el-button v-if="!isHandling && currentStep == 5" @click="reload"> 重新执行 </el-button>
            </div>

        </div>
        <div style="margin: 0 auto; text-align: center;" v-if="!isHandling && currentStep == 5">
            <div style="margin-top:20px">生成结果：（直接点击任意图片进行矢量化）</div>

            <div v-for="char in show_charset">
                <span v-for="i in infer_arr">
                    {{ i }}
                    <img @click="tovector(char, i)"
                        :src="config.request_url + '/file?path=' + text_inversion_dir + '/' + char + '_' + i + '.png'" />
                </span>
                <div>{{ char }}</div>
            </div>

            <el-pagination v-if="!isHandling && currentStep == 5" @current-change="handleCurrentChange"
                :current-page="current_page" :page-size="page_size" layout="total, prev, pager, next"
                :total="charset.length">
            </el-pagination>
        </div>
        <SvgDisplayer v-if="!isHandling && currentStep == 5" ref="svgDisplayer"> </SvgDisplayer>


    </div>
</template>

<script>
// @ is an alias to /src
// import HelloWorld from "@/components/HelloWorld.vue";
import CanvasDrawer from '@/components/canvas-drawer/CanvasDrawer.vue'
import BitMapDisplayer from '@/components/bitmap-displayer/BitmapDisplayer.vue'
import SvgDisplayer from '@/components/svg-displayer/SvgDisplayer';
import CanvasUtil, { blobToBase64Url } from '@/util/CanvasUtil';
import RequestUtil from '@/util/RequestUtil'
const config = require("../config")
const axios = require("axios")
export default {
    name: "HomeView",
    components: {
        CanvasDrawer,
        BitMapDisplayer,
        SvgDisplayer
    },
    data() {
        return {
            config:config,
            isHandling: false,
            currentStep: 0,
            currentTarget: null,
            circles: [],
            progress: 0,
            interval: null,
            show_charset: [],
            charset: [],
            current_page: 1,
            page_size: 1,
            infer_num: 3,
            infer_arr: []
        };
    },
    async mounted() {
        let that = this
        this.config = config
        console.log(config)
        this.getProgress()
    },
    methods: {
        async tovector(char, index) {
            let res = await axios({
                method: "get",
                url: `${config.request_url}/file?path=` + this.text_inversion_dir + `/${char}_${index}.png`,
                responseType: "blob"
            })
            let base64 = await CanvasUtil.blobToBase64Url(res.data)
            this.$refs.svgDisplayer.setBitmap(base64, char)
        },
        handleCurrentChange(val) {
            console.log(`当前页: ${val}`);
            this.current_page = val
            this.show_charset = []
            for (let i = (this.current_page - 1) * this.page_size; i < Math.min(this.current_page * this.page_size, this.charset.length); i++) {
                this.show_charset.push(this.charset[i])
            }
        },
        reload() {
            this.currentStep = 0
            this.progress = 0
        },
        async refresh(){
            let that = this
            let res = await RequestUtil.getCurrentProgress()
            this.isHandling = res.data.data.current_is_handling
            this.currentStep = res.data.data.current_step
            this.progress = parseFloat(res.data.data.progress).toFixed(2)
            if (!this.isHandling && this.currentStep != 0) {
                clearTimeout(this.timer)
                let resultDir = await RequestUtil.getResultDir()
                this.zi2zi_dir = resultDir.data.data.zi2zi_dir
                this.text_inversion_dir = resultDir.data.data.text_inversion_dir
                this.charset = resultDir.data.data.charset.split("")
                this.infer_num = resultDir.data.data.sd_infer_num
                this.infer_arr = []
                console.log(this.infer_num)
                for (let i = 0; i < this.infer_num; i++) this.infer_arr.push(i)
                for (let i = (this.current_page - 1) * this.page_size; i < Math.min(this.current_page * this.page_size, this.charset.length); i++) {
                    this.show_charset.push(this.charset[i])
                }
                return
            }
        },
        async getProgress() {
            let that = this
            let res = await RequestUtil.getCurrentProgress()
            this.isHandling = res.data.data.current_is_handling
            this.currentStep = res.data.data.current_step
            this.progress = parseFloat(res.data.data.progress).toFixed(2)
            if (!this.isHandling && this.currentStep != 0) {
                clearTimeout(this.timer)
                let resultDir = await RequestUtil.getResultDir()
                this.zi2zi_dir = resultDir.data.data.zi2zi_dir
                this.text_inversion_dir = resultDir.data.data.text_inversion_dir
                this.charset = resultDir.data.data.charset.split("")
                this.infer_num = resultDir.data.data.sd_infer_num
                this.infer_arr = []
                console.log(this.infer_num)
                for (let i = 0; i < this.infer_num; i++) this.infer_arr.push(i)
                for (let i = (this.current_page - 1) * this.page_size; i < Math.min(this.current_page * this.page_size, this.charset.length); i++) {
                    this.show_charset.push(this.charset[i])
                }
                return
            }
            this.timer = setTimeout(async () => {
                await that.getProgress()
            }, 15000)
        },
        async submitUpload() {
            this.$refs.upload.submit();
        },
        async uploadSuccess(response, file, fileList) {
            let that = this
            if (response.code == 0) this.$message('上传成功');
            let dir = response.data.dir
            RequestUtil.start(dir)
            this.refresh()
        }
    },
};
</script>

<style scoped>
.home {
    user-select: none;
}

#svg {
    width: 90vw;
    max-width: 500px;
    height: 90vw;
    max-height: 500px;
    border: 1px #0089A7DD solid;
    border-radius: 8px;
    box-shadow: 2px 2px 16px #0089A766;
}

#draw-canvas {
    width: 90vw;
    max-width: 500px;
    height: 90vw;
    max-height: 500px;
    border: 1px #0089A7DD solid;
    border-radius: 8px;
    box-shadow: 2px 2px 16px #0089A766;
    touch-action: none;
    user-select: none;
}

#transfered-canvas {
    width: 90vw;
    max-width: 500px;
    height: 90vw;
    max-height: 500px;
    border: 1px #0089A7DD solid;
    border-radius: 8px;
    box-shadow: 2px 2px 16px #0089A766;
    touch-action: none;
}

.grid-container {
    display: grid;
    margin-top: 20px;
    grid-template-columns: repeat(5, 1fr);
    grid-template-rows: repeat(2, 1fr);
    grid-gap: 10px;
}

img {
    width: 150px;
    height: auto;
}
</style>
