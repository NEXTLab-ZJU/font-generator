class CanvasUtil {
    /**
     * CanvasToBase64Url
     * @param {HTMLCanvasElement} canvas 
     * @returns String Base64Url 
     */
    static canvasToBase64Url(canvas) {
        return canvas.toDataURL("image/png");
    }

    /**
     * CanvasToBlob
     * @param {HTMLCanvasElement} canvas 
     * @returns Promise<Blob> Data
     */
    static canvasToBlob(canvas) {
        return new Promise((resolve, reject) => {
            canvas.toBlob(function (blob) {
                resolve(blob)
            })
        })
    }

    /** 
     * BlobToBase64Url
     * @param {Blob} blob 
     * @returns Promise<String>
     */
    static blobToBase64Url(blob) {
        return new Promise((resolve, reject) => {
            let reader = new FileReader()
            reader.readAsDataURL(blob)
            reader.onload = function (e) {
                resolve(reader.result)
            }
        })
    }

    /**
     * DrawBlobToCanvas
     * @param {Blob} blob 
     * @param {HTMLCanvasElement} canvas 
     * @returns Promise<None>
     */
    static drawBlobToCanvas(blob, canvas) {
        return new Promise(async (resolve, reject) => {
            let dataUrl = await CanvasUtil.blobToBase64Url(blob)
            let img = new Image()
            img.src = dataUrl
            if (!canvas.getContext) {
                reject('浏览器不支持canvas')
            }
            let ctx = canvas.getContext('2d')
            img.onload = function () {

                ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
                resolve()
            }
        })
    }

    /**
     * DrawBase64ToCanvas
     * @param {String} base64 
     * @param {HTMLCanvasElement} canvas 
     * @returns Promise<None>
     */
    static drawBase64ToCanvas(base64, canvas) {
        return new Promise(async (resolve, reject) => {
            let img = new Image()
            img.src = base64
            if (!canvas.getContext) {
                reject('浏览器不支持canvas')
            }
            let ctx = canvas.getContext('2d')
            console.log(img)
            img.onload = function () {
                console.log("onload", base64)
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
                resolve()
            }
        })
    }
}

module.exports = CanvasUtil