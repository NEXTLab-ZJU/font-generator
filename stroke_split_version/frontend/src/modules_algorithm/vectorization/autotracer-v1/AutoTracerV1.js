const CanvasUtil = require("@/util/CanvasUtil");
const VectorizationBase = require("../VectorizationBase");
const { traceImage,simplifyTrace,polygon2bezier} = require("./trace-util")
class AutoTracerV1 extends VectorizationBase{
    constructor(traceLevel = 0.5,simplification = 0.1,angleLimit = 135,shortestLine = 20,fitError = 20){
        super()
        this._traceLevel = traceLevel
        this._simplification = simplification
        this._angleLimit = angleLimit
        this._shortestLine = shortestLine
        this._fitError = fitError
        this._img = null
    }

    /**
     * 
     * @param {Blob} blob  Canvas图片blob数据 
     * @returns {Object} 矢量化后相关信息：{ path , points}
     */
    async generateVectorizationSvg(data) {
        await this.setVectorizationImage(data)
        this._updateGenerate(0)
        return this.getVectorizationResult()
    }

    getVectorizationResult(){
        return {
            polygonPoints:this._polygonPoints,
            bezierPoints:this._bezierPoints,
            polygonSvgPath:this._polygonSvgPath,
            bezierSvgPath:this._bezierSvgPath
        }
    }

    setVectorizationImage(data){
        //get type of variable data
        return new Promise(async (resolve,reject)=>{
            let img = new Image()
            let src 
            if(typeof data == "blob"){
                src = await CanvasUtil.blobToBase64Url(data)
            }else if(typeof data == "string"){
                src = data
            }
            img.src = src
            this._img = img
            img.onload = function(){
                resolve()
            }
        })
        
    }

    // Not Common To All Vectorization Algorithms
    
    setTraceLevel(traceLevel){
        this._traceLevel = traceLevel
        this._updateGenerate(0)
    }

    setSimplification(simplification){
        this._simplification = simplification
        this._updateGenerate(1)
    }

    setAngleLimit(angleLimit){
        this._angleLimit = angleLimit
        this._updateGenerate(2)
    }

    setShortestLine(shortestLine){
        this._shortestLine = shortestLine   
        this._updateGenerate(2)
    }

    setFitError(fitError){
        this._fitError = fitError
        this._updateGenerate(2)
    }

    _updateGenerate(startStep = 0){
        if(!this._img) return
        if (startStep == 0){
            this._updateRoughTrace()
        }
        if(startStep <= 1){
            this._updatePolygonPoints()
        }
        this._updatePolygonPoints()
        this._updateBezierPoints()
        this._updateSvgPath()
    }

    _updateRoughTrace(){
        this._roughTrace = traceImage(this._img, this._traceLevel, true)
    }

    _updatePolygonPoints(){
        console.log("updatePolygonPoints")
        this._polygonPoints = simplifyTrace(this._roughTrace, this._simplification)
    }

    _updateBezierPoints(){
        console.log("updateBezierPoints")
        this._bezierPoints = this._polygonPoints.map(c => polygon2bezier(c, {
            angleLimit: this._angleLimit,
            shortestLine: this._shortestLine,
            fitError: this._fitError
        }))
    }

    _updateSvgPath(){
        this._polygonSvgPath = this._mapPolygonPointsToSvgPath(this._polygonPoints)
        this._bezierSvgPath = this._mapBezierPointsToSvgPath(this._bezierPoints)
    }

    _mapPolygonPointsToSvgPath(polygonData){
        return polygonData.map(path => {
            let pathD = path.map(pt => `${pt[0]},${pt[1]}`).join(' L ')
            return 'M' + pathD + ' Z'
        }).join(' ')
    }

    _mapBezierPointsToSvgPath(bezierPoints){
        return bezierPoints.map(c => {
            let result = `M ${c[0].x},${c[0].y}`
            let ends = true
            c.slice(1).forEach(pt => {
                result += ' '
                if (pt.on) {
                    result += (ends ? 'L ' : ' ') + `${pt.x},${pt.y}`
                    ends = true
                } else {
                    result += (ends ? 'C ' : ' ') + `${pt.x},${pt.y}`
                    ends = false
                }
            })
            result += (ends ? 'L' : ' ') + `${c[0].x},${c[0].y}`
            return result
        }).join(' ')
    }
}


module.exports = AutoTracerV1