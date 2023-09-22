/*
 (c) 2023, Lvkesheng Shen
 CanvasDrawer.js, a high-performance canvas drawer with applepencil or other writting equipment. 
 Support Expressure & tiltX/Y
*/

/**
 * Some Coding Rules:
 * 1. Everything in debugMode should written in function name startsWith "_debug" prefix, and put them in the end of the class.
 * 2. Every inner function should be written after the outter function, and startsWith "_". (inner function means those functions will not be called by other Class).
 */

const config = require("./config")
const CommonUtil = require("../../util/CommonUtil")
const CanvasUtil = require("../../util/CanvasUtil")
class CanvasDrawer {
    /** @param { string | HTMLCanvasElement } canvas */
    /** @param { number } maxStrokeWidth */
    constructor(canvas, maxStrokeWidth = 25, debugMode = false, debugHelper = {}, backgroundCanvas = null) {
        console.log("CanvasDrawer Init")
        // Set Canvas And Some Attributes
        if (typeof (canvas) === 'string') {
            this._canvas = document.getElementById(canvas)
        } else {
            this._canvas = canvas
        }

        if (typeof (backgroundCanvas) === 'string') {
            this._backgroundCanvas = document.getElementById(backgroundCanvas)
        } else {
            this._backgroundCanvas = backgroundCanvas
        }

        this._canvasWidth = this._canvas.width
        this._canvasHeight = this._canvas.height
        this._ctx = this._canvas.getContext('2d')
        this._backgroundCtx = this._backgroundCanvas.getContext('2d')

        // Set Ctx And Some Attributes
        this._maxStrokeWidth = maxStrokeWidth
        this._currentStrokeWidth = this._ctx.lineWidth = maxStrokeWidth / 2
        this._currentColor = this._ctx.strokeStyle = '#000'
        this._backgroundColor = this._ctx.fillStyle = '#fff'
        this._ctx.lineJoin = 'round'
        this._ctx.lineCap = 'round'

        /** @private @type { { path: {x:number, y:number}, originStrokeWidth: number, smoothedStrokeWidth: number, color: string }[] } */
        this._drawnPaths = []
        this._drawing = false

        // BindPointerEvents 
        this._bindPointerEvents()

        // Debug Exclusive Part
        this._debugMode = debugMode
        this._debugHelper = debugHelper


    }

    async drawDataBase64ToCanvas(base64) {
        this._backgroundBase64 = base64
        await CanvasUtil.drawBase64ToCanvas(this._backgroundBase64, this._backgroundCanvas)
    }

    /**
     * getDataBase64Url
     * @returns String Base64Url
     */
    async getDataBase64Url() {
        await this._render(this._ctx, true)
        return CanvasUtil.canvasToBase64Url(this._canvas)
    }

    /**
     * getBlob
     * @returns Promise<Blob>
     */
    async getBlob() {
        await this._render(this._ctx, true)
        return CanvasUtil.canvasToBlob(this._canvas)
    }

    /**
     * setMaxStrokeWidth
     * @param {number} strokeWidth : 40
     */
    setMaxStrokeWidth(strokeWidth) {
        this._maxStrokeWidth = strokeWidth
    }

    /**
     * setStrokeColor
     * @param {String} strokeColor : "#000000"
     */
    setStrokeColor(strokeColor) {
        this._currentColor = this._ctx.strokeStyle = strokeColor
    }

    undoOperation() {
        const popped = this._drawnPaths.pop()
        /** to handle those click event,because those click will cause element with path.length = 0 */
        if (popped !== undefined && popped.path.length === 0) {
            this._drawnPaths.pop()
        }
        this._render()
    }

    clearCanvas() {
        this._drawnPaths = []
        this._render()
    }

    /** 
     * Get the x and y coordinates (on canvas) of a pointer event
     * @param { PointerEvent | MouseEvent | TouchEvent } event
     * @returns { Object { x:number, y:number} } [ x, y ] */
    _getXY(event) {
        // console.log("typeof TouchEvent",typeof TouchEvent)
        // console.log("event instanceof TouchEvent",event instanceof TouchEvent)
        const clientX = typeof TouchEvent !== 'undefined' &&
            event instanceof TouchEvent ?
            event.touches[0].clientX : event.clientX
        const clientY = typeof TouchEvent !== 'undefined' &&
            event instanceof TouchEvent ?
            event.touches[0].clientY : event.clientY
        const rect = this._canvas.getBoundingClientRect()
        const x = (clientX - rect.left) / rect.width * this._canvasWidth
        const y = (clientY - rect.top) / rect.height * this._canvasHeight
        return { x, y }
    }

    _bindPointerEvents() {
        let that = this
        let allowMouse = true
        let timer
        /** @param { MouseEvent | TouchEvent | PointerEvent } e 
         * @param { (any) => any } callback 
         * */
        function handleEvent(e, callback) {
            that._debugHandleChange(e)
            // console.log("e instanceof TouchEvent", e instanceof TouchEvent)
            // console.log("e instanceof MouseEvent", e instanceof MouseEvent)
            if (typeof TouchEvent !== 'undefined' && e instanceof TouchEvent) {
                allowMouse = false
                clearTimeout(timer)
                timer = setTimeout(() => allowMouse = true, 100)
                callback()
            } else if (e instanceof MouseEvent && allowMouse) {
                callback()
            }
        }

        const isPinch = (e) => typeof TouchEvent !== 'undefined' &&
            e instanceof TouchEvent && e.touches.length > 1

        const mouseDrawBegins = e => {
            console.log("onpointer down")
            handleEvent(e, () => {
                this._beginDrawing();
                this._render()
            })
        }

        const drawContinues = e => {
            handleEvent(e, () => {
                if (!isPinch(e)) {
                    if (e.cancelable) e.preventDefault()
                    this._continueDrawing(e)
                }
            })
        }

        const drawEnds = e => {
            console.log("onpointer end")
            handleEvent(e, () => {
                this._endDrawing();
                this._render()
            })
        }

        this._canvas.onpointerdown = mouseDrawBegins;

        this._canvas.onpointermove = drawContinues;

        this._canvas.onpointerup = drawEnds;
        // this._canvas.onpointerout = drawEnds;
        // this._canvas.onpointerleave = drawEnds;

        // Double fingure undo
        // this._canvas.addEventListener('touchstart', e => {
        //     if (e.touches.length === 2) {
        //         this._readyToUndo = true
        //     }
        // })

        // this._canvas.addEventListener('touchmove', e => {
        //     this._readyToUndo = false
        // })

        // this._canvas.addEventListener('touchend', e => {
        //     if (this._readyToUndo) {
        //         this.undo();
        //         this._readyToUndo = false
        //     }
        // })

        // this._canvas.addEventListener('touchcancel', e => {
        //     this._readyToUndo = false
        // })
    }

    _beginDrawing() {
        if (!this._drawing) {
            if (!(this._drawnPaths.length > 0 &&
                this._drawnPaths[this._drawnPaths.length - 1].path.length === 0)) {
                this._drawnPaths.push({ path: [], originStrokeWidth: [], smoothedStrokeWidth: [], color: this._currentColor })
            }
            this._drawing = true
        }
    }

    _continueDrawing(event) {
        if (this._drawing) {
            this._drawnPaths[this._drawnPaths.length - 1].path.push(this._getXY(event))
            this._drawnPaths[this._drawnPaths.length - 1].originStrokeWidth.push(event.pressure * this._maxStrokeWidth)
            this._drawnPaths[this._drawnPaths.length - 1].smoothedStrokeWidth = CommonUtil.calculateMovingAverage(
                this._drawnPaths[this._drawnPaths.length - 1].originStrokeWidth,
                config.SMOOTH_WINDOW_SIZE)
            this._render()
        }
    }

    _endDrawing() {
        if (this._drawing) {
            this._drawing = false
        }
        this._render()
    }

    _setStyle({
        ctx = this._ctx,
        strokeWidth = this._currentStrokeWidth,
        color = this._currentColor
    } = {}) {
        ctx.lineWidth = strokeWidth
        ctx.strokeStyle = color
    }

    async _render(ctx = this._ctx, special = false) {
        var w = this._canvas.width;
        var h = this._canvas.height;
        if (special) {
            ctx.fillRect(0, 0, w, h)
            if (this._backgroundBase64) await CanvasUtil.drawBase64ToCanvas(this._backgroundBase64, this._canvas)
        } else {
            ctx.clearRect(0, 0, w, h);
        }
        this._drawnPaths.forEach(async pathObj => {
            let { path, smoothedStrokeWidth, color } = pathObj
            for (let i = 0; i < path.length; i++) {
                const point = path[i];
                if (i == 0) continue;
                const lastPoint = path[i - 1];
                const xc = (point.x + lastPoint.x) / 2
                const yc = (point.y + lastPoint.y) / 2
                if (i == 1) {
                    ctx.beginPath()
                    ctx.moveTo(xc, yc)
                    continue
                }
                this.setStrokeColor(color)
                ctx.lineWidth = smoothedStrokeWidth[i - 1];
                ctx.quadraticCurveTo(lastPoint.x, lastPoint.y, xc, yc)
                ctx.stroke()
                ctx.closePath()
                ctx.beginPath()
                ctx.moveTo(xc, yc)
            }
            ctx.closePath()
        })
    }

    _debugHandleChange(e) {
        if (this.debugMode) {
            if (this._debugHelper.currentStrokeWidth)
                this._debugHelper.currentStrokeWidth = e.pressure * this._maxStrokeWidth
            if (this._debugHelper.pointerType)
                this._debugHelper.pointerType.innerText = `pointerType:${e.pointerType}`
            if (this._debugHelper.pressure)
                this._debugHelper.pressure.innerText = `Pressure:${e.pressure}`
            if (this._debugHelper.tiltX)
                this._debugHelper.tiltX.innerText = `tiltX:${e.tiltX}`
            if (this._debugHelper.tiltY)
                this._debugHelper.tiltY.innerText = `tiltY:${e.tiltY}`
            if (this._debugHelper.width)
                this._debugHelper.width.innerText = `width:${e.width}`
            if (this._debugHelper.height)
                this._debugHelper.height.innerText = `height:${e.height}`
        }
    }
}

module.exports = CanvasDrawer