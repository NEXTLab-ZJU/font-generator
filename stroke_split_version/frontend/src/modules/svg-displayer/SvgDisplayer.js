class SvgDisplayer {
    constructor(svg) {
        if (typeof (svg) === 'string') {
            /** @type { HTMLOrSVGElement } */
            this._svg = document.getElementById(svg)
        } else {
            this._svg = svg
        }

        this._svgPath = null
    }

    showSvgPathImage(path) {

    }
}


module.exports = SvgDisplayer