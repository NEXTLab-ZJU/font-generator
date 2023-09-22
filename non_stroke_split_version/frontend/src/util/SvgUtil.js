class SvgUtil {
    /**
    @param {String} svgPath - Path In Svg
    @param {Object} viewBox - .x .y .width .height
    @param {String} fileName- fileName like 'bazier',so output will be bazier.svg
    @param {String} color- default to be '#000000'
    */
    static downloadSvg(svgPath, viewBox, fileName, color = "#000000") {
        let file_txt = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${viewBox.x} ${viewBox.y} ${viewBox.width} ${viewBox.height}">\n`
            + `<path d="${svgPath}Z" fill="${color}"/>\n</svg>`
        let txtFile = new Blob([file_txt], { type: ' text/plain' })
        downFile(txtFile, `${fileName}.svg`)
        function downFile(blob, fileName) {
            const link = document.createElement('tmp')
            link.href = window.URL.createObjectURL(blob)
            link.download = fileName
            link.click()
            window.URL.revokeObjectURL(link.href)
        }
    }
}