<template>
    <div style="text-align: center;justify-content: center;">
        <div> 请调整矢量化算法参数以选出最合适的矢量化结果</div>
        <img id="bitmap-img" />
        <svg id="svg-displayer" viewBox="0 0 512 512">
            <g id="bezier-on-points" fill="none" stroke="#000" stroke-width="2"></g>
            <g id="bezier-off-points" fill="#000" stroke="none"></g>
        </svg>
        <div style="width:40%;margin-left:28%;">
            <el-form ref="form" label-width="120px">
                <el-form-item label="traceLevel">
                    <el-slider v-model="traceLevel" :min="0.1" :max="0.9" :step="0.01" @input="setTraceLevel"></el-slider>
                </el-form-item>
                <el-form-item label="simplification">
                    <el-slider v-model="simplification" :min="0.1" :max="0.5" :step="0.01"
                        @input="setSimplification"></el-slider>
                </el-form-item>
                <el-form-item label="angleLimit">
                    <el-slider v-model="angleLimit" :min="90" :max="180" :step="1" @input="setAngleLimit"></el-slider>
                </el-form-item>
                <el-form-item label="shortestLine">
                    <el-slider v-model="shortestLine" :min="0" :max="200" :step="1" @input="setShortestLine"></el-slider>
                </el-form-item>
                <el-form-item label="fitError">
                    <el-slider v-model="fitError" :min="1" :max="30" :step="1" @input="setFitError"></el-slider>
                </el-form-item>
            </el-form>
        </div>
        <el-button type="primary" @click="downloadSvg">保存SVG</el-button>
    </div>
</template>
<script>

import SvgDisPlayer from '@/modules/svg-displayer/SvgDisplayer'
import AutoTracerV1 from '@/modules_algorithm/vectorization/autotracer-v1/AutoTracerV1'
export default {
    props: {

    },
    data() {
        return {
            svgDisPlayer: [],
            vectorizationModel: null,
            transferedBase64Url: null,
            radio: 0,
            traceLevel: 0.5,
            simplification: 0.1,
            angleLimit: 135,
            shortestLine: 20,
            fitError: 20,
            loading: false,
            name:''
        }
    },
    mounted() {
        this.svgDisPlayer = new SvgDisPlayer('svg-displayer')
        // Just need to modify this, To use your own model which is a subclass of StyleTransferBase
        this.vectorizationModel = new AutoTracerV1(
            this.traceLevel,
            this.simplification,
            this.angleLimit,
            this.shortestLine,
            this.fitError
        )
    },
    methods: {
        async getSvgPath() {
            return {
                bezierSvgPath: this.bezierSvgPath,
                bezierPoints: this.bezierPoints
            }
        },
        downloadSvg() {
            let file_txt = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000">\n`
                + `<path d="${this.bezierSvgPath}" fill="#000000"/>\n</svg>`
            let txtFile = new Blob([file_txt], { type: ' text/plain' })
            downFile(txtFile, `${this.name}.svg`)
            function downFile(blob, fileName) {
                const link = document.createElement('a')
                link.href = window.URL.createObjectURL(blob)
                link.download = fileName
                link.click()
                window.URL.revokeObjectURL(link.href)
            }
        },
        async setBitmap(data,name) {
            let {
                polygonPoints,
                bezierPoints,
                polygonSvgPath,
                bezierSvgPath
            } = await this.vectorizationModel.generateVectorizationSvg(data)
            this.name = name
            //draw base64 image to svg
            document.getElementById('bitmap-img').setAttribute('src', data)
            console.log("bezierSvgPath", bezierSvgPath)
            this.transferedBase64Url = data
            this.bezierSvgPath = bezierSvgPath
            this.bezierPoints = bezierPoints
            const svg = document.getElementById('svg-displayer')
            let path = document.getElementById('bezierPath')
            if(!path) path = document.createElementNS("http://www.w3.org/2000/svg", "path");
            
            path.setAttribute("id", 'bezierPath')
            path.setAttribute("d", bezierSvgPath); // 创建一条对角线
            path.setAttribute("stroke", "black");
            path.setAttribute("stroke-width", "1");
            path.setAttribute("fill", "none");
            // 将 <path> 元素添加到 SVG 元素中
            svg.appendChild(path);

            let onPoints = bezierPoints.map(
                c => c.filter(pt => pt.on).map(({ x, y }) => [x, y])).flat()

            let offPoints = bezierPoints.map(
                c => c.filter(pt => !pt.on).map(({ x, y }) => [x, y])).flat()
            updatePointHighlights('bezier-on-points', onPoints, 3)
            updatePointHighlights('bezier-off-points', offPoints, 1.5)
            function updatePointHighlights(svgGroupName, pts, radius = 2) {
                // Updating point visualization
                const group = document.getElementById(svgGroupName)
                group.innerHTML = ''
                pts.forEach(pt => {
                    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle')
                    circle.setAttribute('cx', pt[0])
                    circle.setAttribute('cy', pt[1])
                    circle.setAttribute('r', radius)
                    group.appendChild(circle)
                })
            }
        },
        async setTraceLevel(value) {
            this.traceLevel = value
            this.vectorizationModel.setTraceLevel(value)
            await this.updateSvg()
        },
        async setSimplification(value) {
            this.simplification = value
            this.vectorizationModel.setSimplification(value)
            await this.updateSvg()
        },
        async setAngleLimit(value) {
            this.angleLimit = value
            this.vectorizationModel.setAngleLimit(value)
            await this.updateSvg()
        },
        async setShortestLine(value) {
            this.shortestLine = value
            this.vectorizationModel.setShortestLine(value)
            await this.updateSvg()
        },
        async setFitError(value) {
            this.fitError = value
            this.vectorizationModel.setFitError(value)
            await this.updateSvg()
        },
        async updateSvg() {
            let {
                polygonPoints,
                bezierPoints,
                polygonSvgPath,
                bezierSvgPath
            } = await this.vectorizationModel.getVectorizationResult(this.transferedBase64Url)
            this.bezierSvgPath = bezierSvgPath
            this.bezierPoints = bezierPoints
            const path = document.getElementById('bezierPath')
            if (!path) return
            path.setAttribute("d", bezierSvgPath)


            let onPoints = bezierPoints.map(
                c => c.filter(pt => pt.on).map(({ x, y }) => [x, y])).flat()

            let offPoints = bezierPoints.map(
                c => c.filter(pt => !pt.on).map(({ x, y }) => [x, y])).flat()
            updatePointHighlights('bezier-on-points', onPoints, 3)
            updatePointHighlights('bezier-off-points', offPoints, 1.5)
            function updatePointHighlights(svgGroupName, pts, radius = 2) {
                // Updating point visualization
                const group = document.getElementById(svgGroupName)
                group.innerHTML = ''
                pts.forEach(pt => {
                    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle')
                    circle.setAttribute('cx', pt[0])
                    circle.setAttribute('cy', pt[1])
                    circle.setAttribute('r', radius)
                    group.appendChild(circle)
                })
            }
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

#bitmap-img {
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

#svg-displayer {
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
