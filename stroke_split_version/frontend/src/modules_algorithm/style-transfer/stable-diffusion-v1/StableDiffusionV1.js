const axios = require("axios")
const config = require("../../../config")
const StyleTransferBase = require("../StyleTransferBase")
class StableDiffusionV1 extends StyleTransferBase {
    constructor(prompt, guidance_scale = 7.5, strength = 0.65, num_inference_step = 20, batch_size = 3) {
        super()
        this._prompt = prompt
        this._guidance_scale = guidance_scale
        this._strength = strength
        this._num_inference_step = num_inference_step
        this._batch_size = batch_size
    }

    /**
     * 
     * @param {Blob} blob  Canvas图片blob数据 
     * @returns {Array<Blob>} 生成的图片的blob数组
     */
    async generateStyleTransferImage(blob) {
        let res = await this._sendTransferRequest(blob)
        let filePaths = res.res_file_name
        console.log("filePaths", filePaths)
        let blobArray = []
        for (let i = 0; i < filePaths.length; i++) {
            let filePath = filePaths[i]
            let res = await axios({
                method: "get",
                url: `${config.request_url}/getfile?relative_path=` + filePath,
                responseType: "blob"
            })
            blobArray.push(res.data)
        }
        return blobArray
    }

    setPrompt(prompt) {
        this._prompt = prompt
    }

    setGuidanceScale(guidance_scale) {
        this._guidance_scale = guidance_scale
    }

    setStrength(strength) {
        this._strength = strength
    }

    setNumInferenceStep(num_inference_step) {
        this._num_inference_step = num_inference_step
    }

    setBatchSize(batch_size) {
        this._batch_size = batch_size
    }


    /**
     * 
     * @param {Blob} blob 
     * @returns Array<String> file path array 
     */
    async _sendTransferRequest(blob) {
        let formData = new FormData()
        formData.append('file', blob)
        formData.append('prompt', this._prompt)
        formData.append('guidance_scale', this._guidance_scale)
        formData.append('strength', this._strength)
        formData.append('num_inference_step', this._num_inference_step)
        formData.append('batch_size', this._batch_size)
        const res = await axios({
            method: "post",
            url: `${config.request_url}/transfer`,
            data: formData,
            headers: {
                "Content-Type": "multipart/form-data"
            }
        })

        return res.data
    }
}

module.exports = StableDiffusionV1