const CanvasUtil = require("../../util/CanvasUtil")
class BitmapDisplayer {
    constructor(canvas) {
        if (typeof (canvas) === 'string') {
            /** @type { HTMLCanvasElement } */
            this._canvas = document.getElementById(canvas)
        } else {
            this._canvas = canvas
        }
        this._canvasWidth = this._canvas.width
        this._canvasHeight = this._canvas.height
        this._ctx = this._canvas.getContext('2d')

        this._blobData = null
        this._base64UrlData = null
    }

    showBlobImage(blob) {
        this._blobData = blob
        CanvasUtil.drawBlobToCanvas(blob, this._canvas)
    }

    showBase64UrlImage(base64Url) {
        this._base64UrlData = base64Url
        CanvasUtil.showBase64UrlImage(base64Url)
    }

    getDataBase64Url() {
        return this._base64UrlData ? this._base64UrlData : CanvasUtil.canvasToBase64Url(this._canvas)
    }

    getBlob() {
        return this._blobData ? this._blobData : CanvasUtil.canvasToBlob(this._canvas)
    }

}

module.exports = BitmapDisplayer