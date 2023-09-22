<template>
    <div class="home">
        <el-steps :active="currentStep - 1" finish-status="success" style="width:40%;margin-left:30%;">
            <el-step title="Zi2Zi网络训练"></el-step>
            <el-step title="粗糙位图生成"></el-step>
            <el-step title="SD—TI网络训练"></el-step>
            <el-step title="高质量位图生成"></el-step>
        </el-steps>
        <el-steps :active="currentVectorStep - 1" finish-status="success" style="width:80%;margin-left:10%;">
            <el-step title="矢量化（生成向量矩）"></el-step>
            <el-step title="矢量化（构建笔画数据集）"></el-step>
            <el-step title="矢量化（构建训练集+数据增强）"></el-step>
            <el-step title="矢量化（笔画分割评估训练）"></el-step>
            <el-step title="矢量化（笔画分割推断）"></el-step>
            <el-step title="矢量化（人工挑选）"></el-step>
        </el-steps>

        <div style="margin: 0 auto; text-align: center;margin-top:25px;">
             <el-button @click="refresh" style="margin-bottom: 20px;"> 手动刷新进度（默认每15s自动刷新）</el-button>
            <div v-if="isHandling || isVectorHandling">
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
                        <el-button style="margin-top: 20%;" slot="trigger" size="small" type="primary">选取图像压缩包文件</el-button>
                    </el-upload>
                    <el-upload class="upload-demo" ref="upload2" drag :action="config.request_url + '/upload'" accept="zip"
                        :on-success="uploadSuccess2"
                         :auto-upload="false">
                        <el-button style="margin-top: 20%;" slot="trigger" size="small"
                            type="primary">选取Json压缩包文件</el-button>
                    </el-upload>
                </div>
                <div>
                    <el-button style="margin-left: 10px;" size="small" type="success"
                        @click="submitUpload">上传到服务器</el-button>
                </div>
                <div slot="tip" class="el-upload__tip">只能上传zip文件</div>
            </div>
            <div v-if="!isHandling && !isVectorHandling && currentStep != 0 && currentVectorStep != 0" >
            <div>
                <span v-for="char in show_charset" style="margin-left=10px;">
                    <button @click="changeChar(char)">{{ char }}</button>
                </span>
            </div>
            <el-pagination v-if="!isHandling && currentStep == 5" @current-change="handleCurrentChange"
                :current-page="current_page" :page-size="page_size" layout="total, prev, pager, next"
                :total="charset.length">
            </el-pagination>
            <div>
                <div style="display: inline-block;">
                    <svg id="svg-displayer" viewBox="0 0 512 512">
                    </svg>
                </div>
                <div>
                    <div style="display: inline-block;">
                        <span v-for="i, idx in current_char_json[0]" v-show="idx < 5">
                            <svg @click="changeCurrent(idx)" :id="'svg' + idx" viewBox="0 0 512 512" class="small-svg"> </svg>
                        </span>
                    </div>
                </div>

            </div>

            <button @click="save"> 保存SVG </button>
        </div>
        </div>



    </div>
</template>

<script>
// @ is an alias to /src

import CanvasDrawer from '@/components/canvas-drawer/CanvasDrawer.vue'
import BitMapDisplayer from '@/components/bitmap-displayer/BitmapDisplayer.vue'
import SvgDisplayer from '@/components/svg-displayer/SvgDisplayer';
import CanvasUtil, { blobToBase64Url } from '@/util/CanvasUtil';
import RequestUtil from '@/util/RequestUtil'
const config = require("@/config")
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
            isVectorHandling: false,
            currentStep: 0,
            currentVectorStep: 0,
            currentTarget: null,
            circles: [],
            progress: 0,
            interval: null,
            show_charset: [],
            charset: [],
            current_page: 1,
            page_size: 10,
            pic_dir: '',
            json_dir: '',
            stroke_json_dir: '',
            current_char_json: {
                "0": [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
            },
            current_char_chosen: {
                "0": 0
            },
            current_key: "0",
            current_char: '',
            show_beixuan_number:5
        };
    },
    async mounted() {
        let that = this
        this.config = config
        this.getProgress()
    },
    methods: {
        async tovector(char, index) {
            let res = await axios({
                method: "get",
                url: `${config.request_url}/ti/` + char + `_${index}.png`,
                responseType: "blob"
            })
            let base64 = await CanvasUtil.blobToBase64Url(res.data)
            this.$refs.svgDisplayer.setBitmap(base64, char)
        },
        save() {
            const svg = document.getElementById("svg-displayer");
            let path = ""
            for (let i = 0; i < svg.childNodes.length; i++) {
                path += svg.childNodes[i].getAttribute('d') + "Z "
            }
            let file_txt = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">\n`
                + `<path d="${path}" fill="#000000"/>\n</svg>`
            let txtFile = new Blob([file_txt], { type: ' text/plain' })
            downFile(txtFile, `${this.current_char}.svg`)
            function downFile(blob, fileName) {
                const link = document.createElement('a')
                link.href = window.URL.createObjectURL(blob)
                link.download = fileName
                link.click()
                window.URL.revokeObjectURL(link.href)
            }
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
            this.progress = res.data.data.progress
            console.log("getProgress", this.isHandling)
            if (!this.isHandling && this.currentStep != 0) {
                console.log("finish")
                console.log("getCurrentVectorProgress")
                let that = this
                let res = await RequestUtil.getCurrentVectorProgress()
                this.isVectorHandling = res.data.data.current_is_handling
                this.currentVectorStep = res.data.data.current_step
                this.progress = res.data.data.progress
                if (!this.isVectorHandling && this.currentVectorStep != 0) {
                    clearTimeout(this.timer)
                    let resultDir = await RequestUtil.getResultDir()
                    this.zi2zi_dir = resultDir.data.data.zi2zi_dir
                    this.text_inversion_dir = resultDir.data.data.text_inversion_dir
                    this.charset = resultDir.data.data.charset.split("")
                    console.log("this.charset", this.charset)
                    this.stroke_json_dir = resultDir.data.data.stroke_json_dir
                    for (let i = (this.current_page - 1) * this.page_size; i < Math.min(this.current_page * this.page_size, this.charset.length); i++) {
                        this.show_charset.push(this.charset[i])
                    }
                    this.current_char = this.show_charset[0]
                    this.test()
                }
            }
        },
        async getProgress() {
            let that = this
            let res = await RequestUtil.getCurrentProgress()
            this.isHandling = res.data.data.current_is_handling
            this.currentStep = res.data.data.current_step
            this.progress = res.data.data.progress
            console.log("getProgress", this.isHandling)
            if (!this.isHandling && this.currentStep != 0) {
                console.log("finish")
                clearTimeout(this.timer)
                this.getCurrentVectorProgress()
                return
            }
            this.timer = setTimeout(async () => {
                await that.getProgress()
            }, 15000)
        },
        async getCurrentVectorProgress() {
            console.log("getCurrentVectorProgress")
            let that = this
            let res = await RequestUtil.getCurrentVectorProgress()
            this.isVectorHandling = res.data.data.current_is_handling
            this.currentVectorStep = res.data.data.current_step
            this.progress = res.data.data.progress
            if (!this.isVectorHandling && this.currentVectorStep != 0) {
                clearTimeout(this.timer)
                let resultDir = await RequestUtil.getResultDir()
                this.zi2zi_dir = resultDir.data.data.zi2zi_dir
                this.text_inversion_dir = resultDir.data.data.text_inversion_dir
                this.charset = resultDir.data.data.charset.split("")
                console.log("this.charset", this.charset)
                this.stroke_json_dir = resultDir.data.data.stroke_json_dir
                for (let i = (this.current_page - 1) * this.page_size; i < Math.min(this.current_page * this.page_size, this.charset.length); i++) {
                    this.show_charset.push(this.charset[i])
                }
                this.current_char = this.show_charset[0]
                this.test()
                return
            }
            this.timer = setTimeout(async () => {
                await that.getCurrentVectorProgress()
            }, 15000)
        },
        changeChar(char) {
            this.current_char = char
            for (let i = 0; i < this.current_char_json[this.current_key].length; i++) {
                const svg = document.getElementById(`svg${i}`);
                while (svg.childNodes.length != 0) {
                    svg.removeChild(svg.childNodes[0])
                }
            }
            this.test()
        },
        async submitUpload() {
            this.$refs.upload.submit();
            this.$refs.upload2.submit();
        },
        async uploadSuccess(response, file, fileList) {
            let that = this
            if (response.code == 0) this.$message('上传成功');
            this.pic_dir = response.data.dir
            if (this.json_dir != "") {
                RequestUtil.start(this.pic_dir, this.json_dir)
                this.refresh()
            }
        },
        async uploadSuccess2(response, file, fileList) {
            let that = this
            if (response.code == 0) this.$message('上传成功');
            this.json_dir = response.data.dir
            if (this.pic_dir != "") {
                RequestUtil.start(this.pic_dir, this.json_dir)
                this.refresh()
            }
        },
        async test() {
            let stroke_json_dir = this.stroke_json_dir
            let char = this.current_char
            let res = await axios({
                method: "get",
                url: `${config.request_url}/json_file?path=` + stroke_json_dir + `/${char}.json`,
                responseType: "json"
            })
            this.current_char_json = res.data
            for (let key in this.current_char_json) {
                this.current_char_chosen[key] = 0
            }
            console.log("this.current_char_json", this.current_char_json)
            this.draw()
        },
        draw() {
            let that = this
            const svg = document.getElementById("svg-displayer");
            while (svg.childNodes.length != 0) {
                svg.removeChild(svg.childNodes[0])
            }
            for (let key in this.current_char_json) {
                let coutour = this.current_char_json[key][this.current_char_chosen[key]]
                let path = this.convert(coutour)
                let ele = document.createElementNS("http://www.w3.org/2000/svg", "path");
                ele.setAttributeNS(null, "d", path);
                if (key == this.current_key) {
                    ele.setAttributeNS(null, "fill", "blue");
                }
                ele.setAttributeNS(null, "id", key)
                const svg = document.getElementById("svg-displayer");
                svg.appendChild(ele);
                ele.addEventListener('click', function (e) {
                    console.log('click', e.target.id)
                    that.current_key = e.target.id
                    that.draw()
                })
            }
            
            for (let i = 0; i < this.current_char_json[this.current_key].length; i++) {
                const svg = document.getElementById(`svg${i}`);
                for (let key in this.current_char_json) {
                    let coutour = this.current_char_json[key][i]
                    let path = this.convert(coutour)
                    let ele = document.createElementNS("http://www.w3.org/2000/svg", "path");
                    ele.setAttributeNS(null, "d", path);
                    if (key == this.current_key) {
                        ele.setAttributeNS(null, "fill", "red");
                    }
                    if(i == this.current_char_chosen[key] && key == this.current_key){
                        ele.setAttributeNS(null, "fill", "blue")
                    }
                    ele.setAttributeNS(null, "id", key)
                    svg.appendChild(ele);
                    ele.addEventListener('click', function (e) {
                        that.current_key = e.target.id
                        that.draw()
                    })
                }
                // let coutour = this.current_char_json[this.current_key][i]
                // let path = this.convert(coutour)
                // let ele = document.createElementNS("http://www.w3.org/2000/svg", "path");
                // ele.setAttributeNS(null, "d", path);
                
                // svg.appendChild(ele);
            }
        },
        convert(coutour) {
            let points = coutour.contour
            let d = "M " + points[0].x + " " + points[0].y;
            for (let i = 1; i < points.length;) {
                if (points[i].on) {
                    //如果是直线
                    d += " L " + points[i].x + " " + points[i].y
                    i = i + 1
                } else {
                    // 使用三次贝塞尔曲线命令C  
                    if (i + 2 == points.length) {
                        d += " C " + points[i].x + " " + points[i].y + " "
                            + points[i + 1].x + " " + points[i + 1].y + " "
                            + points[0].x + " " + points[0].y;
                    } else {
                        d += " C " + points[i].x + " " + points[i].y + " "
                            + points[i + 1].x + " " + points[i + 1].y + " "
                            + points[i + 2].x + " " + points[i + 2].y;
                    }
                    i = i + 3
                }

            }
            return d
        },
        changeCurrent(idx) {
            this.current_char_chosen[this.current_key] = idx
            console.log("this.current_char_chosen", this.current_char_chosen)
            this.draw()
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
    width: 50vw;
    max-width: 500px;
    height: 50vw;
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

#svg-displayer {
    width: 60vw;
    max-width: 300px;
    height: 606vw;
    max-height: 300px;
    border: 1px #0089A7DD solid;
    border-radius: 8px;
    box-shadow: 2px 2px 16px #0089A766;
    touch-action: none;
    user-select: none;
    margin: 20px;
}

.small-svg {
    width: 10vw;
    max-width: 500px;
    height: 10vw;
    max-height: 500px;
    border: 1px #0089A7DD solid;
    border-radius: 8px;
    box-shadow: 2px 2px 16px #0089A766;
    touch-action: none;
    user-select: none;
    margin: 20px;
}
</style>
