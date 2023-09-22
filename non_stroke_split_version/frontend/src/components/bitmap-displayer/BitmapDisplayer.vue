<template>
    <div style="text-align: center;justify-content: center;">
        <div v-show="!dialogVisible">
            <div> 请从下述生成图像中选择一个你认为最合适的图像</div>
            <div style="display:flex;text-align: center;justify-content: center;flex-wrap: wrap;">
                <div v-for="(i, index) in  batch_size * [1]">
                    <canvas class="bitmap-display-canvas" :id="'bitmap-display-canvas-' + index" width="512"
                        height="512"></canvas>
                    <div>
                        <el-radio v-model="radio" :label="index">生成图{{ i }}</el-radio>
                    </div>
                </div>
            </div>
            <div style="margin-top: 30px;">
                <el-button @click="drawer = true" type="primary">
                    调整参数
                </el-button>
                <el-button type="primary" @click="refreshResult" style="margin-left: 60px;">重新生成</el-button>
                <el-button type="primary" @click="modifyResult" style="margin-left: 60px;"> 绘制修改</el-button>
            </div>
            <el-drawer title="推断选项：" :visible.sync="drawer" direction="ltr">
                <el-form ref="form" label-width="120px">
                    <el-form-item label="推断风格">
                        <el-select style="margin-left:-20%;width:80%;" v-model="prompt" placeholder="请选择">
                            <el-option v-for="item in promptOptions" :key="item.value" :label="item.label"
                                :value="item.value">
                            </el-option>
                        </el-select>
                    </el-form-item>
                    <el-form-item label="推断风格强度">
                        <el-slider style="width:80%;" v-model="guidance_scale" :min="1" :max="20" :step="1"></el-slider>
                    </el-form-item>
                    <el-form-item label="偏离原图强度">
                        <el-slider style="width:80%;" v-model="strength" :min="0" :max="1" :step="0.05"></el-slider>
                    </el-form-item>
                    <el-form-item label="推断轮数">
                        <el-slider style="width:80%;" v-model="num_inference_step" :min="10" :max="50"
                            :step="1"></el-slider>
                    </el-form-item>
                </el-form>
                <el-button type="primary" @click="refreshResult">确认修改</el-button>
            </el-drawer>
        </div>

        <!-- <el-dialog title="提示" :visible.sync="dialogVisible" width="30%" :before-close="handleClose">
            <CanvasDrawer ref="canvasDrawer"> </CanvasDrawer>
        </el-dialog> -->
        <div v-show="dialogVisible">
            <CanvasDrawer ref="canvasDrawer" canvas-id="draw-canvas2" backgroundCanvasId="bg2"> </CanvasDrawer>
            <el-button type="primary" @click="finishModify" style="margin-left: 60px;">完成重绘</el-button>
        </div>

    </div>
</template>
<script>
import CanvasDrawer from '@/components/canvas-drawer/CanvasDrawer.vue'
import BitmapDisplayer from '@/modules/bitmap-displayer/BitmapDisplayer'
import StableDiffusionV1 from '@/modules_algorithm/style-transfer/stable-diffusion-v1/StableDiffusionV1'
export default {
    props: {

    },
    components: {
        CanvasDrawer,
    },
    data() {
        return {
            promptOptions: [
                {
                    label: "开诚",
                    value: "kaicheng"
                },
                {
                    label: "楷体",
                    value: "kaiti"
                },
                {
                    label: "宋体",
                    value: "songti"
                },
            ],
            drawer: false,
            bitmapDisplayer: [],
            styleTransferModel: null,
            drawnBlob: null,
            radio: 0,
            prompt: 'kaicheng',
            guidance_scale: 7.5,
            strength: 0.65,
            num_inference_step: 20,
            batch_size: 3,
            loading: false,
            dialogVisible: false
        }
    },
    mounted() {
        for (let i = 0; i < this.batch_size; i++) {
            this.bitmapDisplayer.push(new BitmapDisplayer(`bitmap-display-canvas-${i}`))
        }
        // Just need to modify this, To use your own model which is a subclass of StyleTransferBase
        this.styleTransferModel = new StableDiffusionV1(
            this.prompt,
            this.guidance_scale,
            this.strength,
            this.num_inference_step,
            this.batch_size
        )
    },
    methods: {
        setDrawnBlob(blob) {
            this.drawnBlob = blob
            this.refreshResult()
        },
        async finishModify() {
            // this.dialogVisible = false
            // this.drawnBlob = await this.$refs.canvasDrawer.getBase64()
            // let test = document.getElementById('test')
            // test.src = this.drawnBlob
            // console.log("this.drawnBlob")
            this.dialogVisible = false
            this.drawnBlob = await this.$refs.canvasDrawer.getCanvasBlob()
            this.refreshResult()
        },
        async refreshResult() {
            if (this.drawer) this.drawer = false
            const loading = this.$loading({
                lock: true,
                text: 'Loading',
                spinner: 'el-icon-loading',
                background: 'rgba(0, 0, 0, 0.7)'
            });
            let blobArray = await this.styleTransferModel.generateStyleTransferImage(this.drawnBlob)
            for (let i = 0; i < this.batch_size; i++) {
                this.bitmapDisplayer[i].showBlobImage(blobArray[i])
            }
            loading.close()
        },
        getSelectedImage() {
            return this.bitmapDisplayer[this.radio].getDataBase64Url()
        },
        async modifyResult() {
            this.dialogVisible = true
            let base64 = this.bitmapDisplayer[this.radio].getDataBase64Url()
            await this.$refs.canvasDrawer.drawDataBase64ToCanvas(base64)
        }
    }
}
</script>
<style scoped>
.block {
    width: 70vw;
    margin-left: 15vw;
    display: flex;
    margin-top: 20px;
}

.bitmap-display-canvas {
    width: 90vw;
    max-width: 500px;
    height: 90vw;
    max-height: 500px;
    border: 1px #0089A7DD solid;
    border-radius: 8px;
    box-shadow: 2px 2px 16px #0089A766;
    touch-action: none;
    user-select: none;
    margin: 20px;
}
</style>
