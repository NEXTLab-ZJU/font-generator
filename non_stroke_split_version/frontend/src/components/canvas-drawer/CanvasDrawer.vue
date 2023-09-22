<template>
    <div style="text-align:center">
        <div> 推荐使用Chrome浏览器</div>
        <div style="margin-bottom:20px"> 请注意：如您使用Apple Pencil作为书写工具，请前往设置->Apple Pencil处，将随手写功能关闭，否则会导致BUG</div>
        <div style="position: relative;margin: 0 auto;width: 512px;height: 512px;">
            <canvas :id="canvasId" width="512" height="512" style="position: absolute; top: 0; left: 0;"></canvas>
            <canvas :id="backgroundCanvasId" width="512" height="512"></canvas>
        </div>

        <div style="text-align:center;justify-content: center;display: flex;">
            <el-slider v-model="maxStrokeWidth" :min="15" :max="50" :step="2" style="width:30vw;"
                @input="changeMaxStrokeWidth">
            </el-slider>
        </div>
        <el-button @click="changeToPen"> 笔工具</el-button>
        <el-button @click="changeToEraser"> 橡皮工具</el-button>
        <el-button @click="clearDrawCanvas" type="primary">
            <i class="fa-solid fa-eraser"></i> 清空
        </el-button>
        <el-button @click="undoDrawCanvas" type="primary">
            <i class="fa-solid fa-rotate-left"></i> 撤销
        </el-button>
    </div>
</template>
<script>


import CanvasDrawer from '@/modules/canvas-drawer/CanvasDrawer'
export default {
    props: {
        coreStep: Number,
        canvasId: String,
        backgroundCanvasId: String
    },
    watch: {
        coreStep(value) {
            if (value == 2) {
                this.$emit('aaa', 'canvas-drawer')
            }
        }
    },
    data() {
        return {
            maxStrokeWidth: 25,
            canvasDrawer: null
        }
    },
    mounted() {
        this.canvasDrawer = new CanvasDrawer(this.canvasId, this.maxStrokeWidth, false, {}, this.backgroundCanvasId)
    },
    methods: {
        changeToPen() {
            this.canvasDrawer.setStrokeColor("#000")
        },
        changeToEraser() {
            this.canvasDrawer.setStrokeColor("#fff")
        },
        changeMaxStrokeWidth(value) {
            this.canvasDrawer.setMaxStrokeWidth(value)
        },
        clearDrawCanvas() {
            this.canvasDrawer.clearCanvas()
        },
        undoDrawCanvas() {
            this.canvasDrawer.undoOperation()
        },
        getCanvasBlob() {
            return this.canvasDrawer.getBlob()
        },
        getBase64() {
            return this.canvasDrawer.getDataBase64Url()
        },
        async drawDataBase64ToCanvas(base64) {
            console.log("base64 1", base64)
            await this.canvasDrawer.drawDataBase64ToCanvas(base64)
        }
    }
}
</script>
<style scoped>
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
</style>
