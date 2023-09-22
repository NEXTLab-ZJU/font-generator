class VectorizationBase {
    constructor() {
        if (new.target === VectorizationBase) {
            throw new Error('VectorizationBase为抽象类，不能被实例化');
        }
    }

    /**
     * 
     * @param {Blob} blob  Canvas图片blob数据 
     * @returns {Object} 矢量化后相关信息：{ path , points}
     */
    generateVectorizationSvg(blob) {
        throw '"' + this.constructor.name + "'类没有generateVectorizationSvg()方法,请重写实现";
    }

}

module.exports = VectorizationBase