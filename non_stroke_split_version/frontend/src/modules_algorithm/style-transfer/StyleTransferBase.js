class StyleTransferBase {
    constructor() {
        if (new.target === StyleTransferBase) {
            throw new Error('StyleTransferBase为抽象类，不能被实例化');
        }
    }

    /**
     * 
     * @param {Blob} blob  Canvas图片blob数据 
     * @returns {Array<Blob>} 生成的图片的blob数组
     */
    generateStyleTransferImage(blob) {
        throw '"' + this.constructor.name + "'类没有generateStyleTransferImage()方法,请重写实现";
    }

}

module.exports = StyleTransferBase