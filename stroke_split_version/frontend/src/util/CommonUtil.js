class CommonUtil {
    static calculateMovingAverage(arr, windowSize = 5) {
        if(arr.length < windowSize) return arr

        const result = [];
        
        let beforeElementNumber = Math.ceil(windowSize/2) - 1
        let afterElementNumber = Math.floor(windowSize/2)
        for (let i = 0; i < beforeElementNumber; i++){
            result.push(arr[i])
        }
        let sum = 0
        for(let i = beforeElementNumber;i<(arr.length-afterElementNumber);i++){
            for(let j = -beforeElementNumber;j<=afterElementNumber;j++){
                sum += arr[i+j]
            }
            result.push(sum/windowSize)
            sum = 0
        }
        for (let i = afterElementNumber - 1; i >=0 ; i--){
            result.push(arr[arr.length - i - 1])
        }
        return result;
    }
}



module.exports = CommonUtil