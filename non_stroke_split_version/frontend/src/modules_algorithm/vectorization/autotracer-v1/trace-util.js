
const d3 = require("d3")
const fitCurve = require("./fit-curve")

function getSqDist(p1, p2) {
    var dx = p1[0] - p2[0],
        dy = p1[1] - p2[1];
    return dx * dx + dy * dy;
}

// square distance from a point to a segment
function getSqSegDist(p, p1, p2) {
    var x = p1[0],
        y = p1[1],
        dx = p2[0] - x,
        dy = p2[1] - y;

    if (dx !== 0 || dy !== 0) {

        var t = ((p[0] - x) * dx + (p[1] - y) * dy) / (dx * dx + dy * dy);

        if (t > 1) {
            x = p2[0];
            y = p2[1];

        } else if (t > 0) {
            x += dx * t;
            y += dy * t;
        }
    }

    dx = p[0] - x;
    dy = p[1] - y;

    return dx * dx + dy * dy;
}
// rest of the code doesn't care about point format

// basic distance-based simplification
function simplifyRadialDist(points, sqTolerance) {

    var prevPoint = points[0],
        newPoints = [prevPoint],
        point;

    for (var i = 1, len = points.length; i < len; i++) {
        point = points[i];

        if (getSqDist(point, prevPoint) > sqTolerance) {
            newPoints.push(point);
            prevPoint = point;
        }
    }

    if (prevPoint !== point) newPoints.push(point);

    return newPoints;
}

function simplifyDPStep(points, first, last, sqTolerance, simplified) {
    var maxSqDist = sqTolerance,
        index;

    for (var i = first + 1; i < last; i++) {
        var sqDist = getSqSegDist(points[i], points[first], points[last]);

        if (sqDist > maxSqDist) {
            index = i;
            maxSqDist = sqDist;
        }
    }

    if (maxSqDist > sqTolerance) {
        if (index - first > 1) simplifyDPStep(points, first, index, sqTolerance, simplified);
        simplified.push(points[index]);
        if (last - index > 1) simplifyDPStep(points, index, last, sqTolerance, simplified);
    }
}

// simplification using Ramer-Douglas-Peucker algorithm
function simplifyDouglasPeucker(points, sqTolerance) {
    var last = points.length - 1;

    var simplified = [points[0]];
    simplifyDPStep(points, 0, last, sqTolerance, simplified);
    simplified.push(points[last]);

    return simplified;
}

// both algorithms combined for awesome performance
function simplify(points, tolerance) {
    if (points.length <= 2) return points;

    var sqTolerance = tolerance !== undefined ? tolerance * tolerance : 1;

    points = simplifyRadialDist(points, sqTolerance);
    points = simplifyDouglasPeucker(points, sqTolerance);

    return points;
}

function traceImage(img, traceLevel=0.5, traceBlack=true) {
    let canvas, context
    if (img instanceof HTMLImageElement) {
      if (!img.complete) throw Error('Image to trace has not loaded completely.')
      canvas = document.createElement('canvas')
      context = canvas.getContext('2d')
      canvas.width = img.width
      canvas.height = img.height
      context.drawImage(img, 0, 0)
    } else if ('getContext' in img) {
      canvas = img
      context = canvas.getContext('2d')
    } else {
      throw Error('Unsupported data type.')
    }
    console.log("sss",img.width, img.height)
    const imgData = context.getImageData(0, 0, img.width, img.height)
    console.log("sss2")
    // Converting the image to a 1-D array
    const imgArr = []
    for (let i = 0; i < imgData.width * imgData.height; i++) {
      const grayVal = (imgData.data[i*4] + imgData.data[i*4+1]
        + imgData.data[i*4+2])/3/255
      if (traceBlack) imgArr.push(1 - grayVal)
      else imgArr.push(grayVal)
    }
    // Contour tracing using d3-contours (marching cubes)
    // We make clock-wise as the positive direction, since the zero point is to 
    // the left top of the figure
    let contours = d3.contours().size([img.width, img.height])
      .thresholds([ traceLevel ])(imgArr)[0].coordinates.flat().map(c => c.reverse())
    return contours
  }
  
  // Simplifying the rough trace result
  function simplifyTrace(contours, tolerance=0.2) {
    // PASS 1: Selecting all closed contours
    contours = contours.filter(c => {
      const n = c.length
      if (c.length === 0) return false
      return c[0][0] === c[n-1][0] && c[0][1] === c[n-1][1]
    })
    // PASS 2: Approximate polygon using Ramer-Douglas-Peucker algorithm
    contours = contours.map(c => simplify(c, tolerance).slice(0, -1))
    // PASS 3: Merging points that are too close together (that are under raw
    // tracing precision)
    function clearPoints(c, threshold=2.5) {
      const result = []
      c.forEach(pt => {
        if (result.length === 0) { result.push([ ...c[0] ]); return }
        const last = result[result.length - 1]
        if ((last[0] - pt[0])**2 + (last[1] - pt[1])**2 <= threshold){
          result[result.length - 1] = [
            (last[0] + pt[0])/2, (last[1] + pt[1])/2 ]
        } else {
          result.push([ ...pt ])
        }
      })
      const first = result[0]
      const last = result[result.length - 1]
      if ((first[0] - last[0])**2 + (first[1] - last[1])**2 <= threshold 
      && result.length > 1) {
        result[0] = [
          (first[0] + last[0])/2, (first[1] + last[1])/2 ]
        result.pop()
      }
      return result
    }
    return contours.map(c => clearPoints(c))
  }
  
  /** Project a vector to the direction defined by another vector
   * @param { [ number, number ] } from
   * @param { [ number, number ] } to */
  function projectVector(from, to) {
    if (to[0] === to[1] === 0) return [ 0, 0 ]
    // from.to
    const a_b = from[0] * to[0] + from[1] * to[1]
    // to.to
    const b_b = to[0] * to[0] + to[1] * to[1]
    const r = a_b/b_b
    return [ r*to[0], r*to[1] ]
  }
  
  // polygonData = [[ [ 160, 160 ], [ 200, 100 ], [ 200, 200 ], [ 100, 200 ] ]]
  function differential(pts) {
    return pts.map((pt, i) => {
      const prev = pts[(i+pts.length-1)%pts.length]
      const next = pts[(i+1)%pts.length]
      return [ next[0] - prev[0], next[1] - prev[1] ]
    })
  }
  
  function curvatureRadii(pts) {
    const d1s = differential(pts)
    const d2s = differential(d1s)
    return d1s.map((d1, i) => {
      const d2 = d2s[i]
      const [ x1, y1 ] = d1, [ x2, y2 ] = d2
      return (x1*x1 + y1*y1)**1.5/(x1*y2 - y1*x2)
    })
  } 
  
  function unitNormals(pts) {
    const d1s = differential(pts)
    return d1s.map((d1) => {
      const norm = Math.sqrt(d1[0]*d1[0] + d1[1]*d1[1])
      return [ d1[1]/norm, -d1[0]/norm ]
    })
  }
  
  function unitTangents(pts) {
    const d1s = differential(pts)
    return d1s.map((d1) => {
      const norm = Math.sqrt(d1[0]*d1[0] + d1[1]*d1[1])
      return [ d1[0]/norm, d1[1]/norm ]
    })
  }
  
  /** This function computes the interior corner at a certain point. Negative when
   * the interior angle is reflex. */
  function cornerAngles(pts) {
    return pts.map((pt, i) => {
      const prev = pts[(i+pts.length-1)%pts.length]
      const next = pts[(i+1)%pts.length]
      const x1 = next[0] - pt[0], y1 = next[1] - pt[1]
      const x2 = prev[0] - pt[0], y2 = prev[1] - pt[1]
      return Math.atan2(x1*y2 - y1*x2, x1*x2 + y1*y2)
    })
  }
  
  function findCorners(angles, radii, angleLimit=135, maxAngleCurvature=Infinity) {
    return angles.map((angle, i) => 
      Math.abs(angle * 180 / Math.PI) < angleLimit &&
      Math.abs(radii[i]) <= maxAngleCurvature
    )
  }
  
  function findXExtrema(pts) {
    return pts.map((pt, i) => {
      const prev = pts[(i+pts.length-1)%pts.length]
      const next = pts[(i+1)%pts.length]
      if ((pt[0] - prev[0]) * (pt[0] - next[0]) >= 0) return true
      return false
    })
  }
  
  function findYExtrema(pts) {
    return pts.map((pt, i) => {
      const prev = pts[(i+pts.length-1)%pts.length]
      const next = pts[(i+1)%pts.length]
      if ((pt[1] - prev[1]) * (pt[1] - next[1]) >= 0) return true
      return false
    })
  }
  
  /** This function detects points where the curvature reverts */
  function curvatureReversions(radii) {
    // PASS1: Select all points where the curvature reverts first
    const pass1 = radii.map((r, i) => {
      const next = radii[(i+1)%radii.length]
      return next * r < 0
    })
    // PASS2: Remove contiguous reversions
    const pass2 = pass1.map((reverts, i) => {
      const n = pass1.length
      const prevReverts = pass1[(i-1+n)%n]
      const nextReverts = pass1[(i+1)%n]
      return reverts && !prevReverts && !nextReverts
    })
    return pass2
  }
  
  /** Finding line segments */
  function findLines(pts, corners, shortestLine=20) {
    if (pts.length !== corners.length) 
      throw Error('Incompatible contour length info')
  
    const squaredDists = pts.map((pt, i) => {
      const next = pts[(i+1)%pts.length]
      return (next[0] - pt[0])**2 + (next[1] - pt[1])**2
    })
    // PASS1: Find all line segment starting point
    const pass1 = squaredDists.map(d => d > shortestLine**2)
    // Preparing for PASS2, indices of contiguous line ranges
    const runIndices = []
    pass1.forEach((isLine, i) => {
      if (!isLine) return
      if (runIndices.length === 0) {
        runIndices.push([ i, i ])
        return
      }
      const lastRun = runIndices[runIndices.length-1]
      if (i - lastRun[1] <= 1 && !corners[i]) {
        lastRun[1] = i
      } else {
        runIndices.push([ i, i ])
      }
    })
    // Handling the first run
    const lastRun = runIndices[runIndices.length-1]
    if (runIndices.length !== 0 && !corners[0] && runIndices[0][0] === 0
    && lastRun[1] === pass1.length - 1) {
      runIndices[0][0] = lastRun[0]
      runIndices.pop()
    }
    // PASS2: If two adjacent lines does not make a corner, then the shorter one
    // is not a line
    const pass2 = Array(pts.length).fill(false)
    runIndices.forEach((indiceRange) => {
      const [ lower, upper ] = indiceRange
      let loclDists
      if (lower > upper) {
        loclDists = [ ...squaredDists.slice(lower),
          ...squaredDists.slice(0, upper + 1) ]
      } else {
        loclDists = squaredDists.slice(lower, upper + 1)
      }
      // Local minimum
      const minIndex = loclDists.reduce((mi, val, i, arr) => {
        return val < arr[mi] ? i : mi
      }, 0)
      pass2[(lower + minIndex) % pts.length] = true
    })
    return pass2
  }
  
  /** Segmenting a polygon to pieces that are ready for piece-wise tracing */
  function segmentPolygon(pts, { 
    angleLimit = 120,
    shortestLine = 20
  } = {}) {
    const radii = curvatureRadii(pts)
    const angles = cornerAngles(pts)
    // Booleans
    const corners = findCorners(angles, radii, angleLimit)
    const lines = findLines(pts, corners, shortestLine)
    const reversions = curvatureReversions(radii)
    const xExtremas = findXExtrema(pts)
    const yExtremas = findYExtrema(pts)
    // const tangents = unitTangents(pts)
    /** Segments of the raw polygon
     * @type { { cornerIn: boolean, isLine: boolean, reversion: boolean,
     * xExtrema: boolean, yExtrema: boolean, 
    //  * tangents: [ [ number, number ], [ number, number ] ],
     * indiceRange: [ number, number ] }[] } */
    const segments = []
    pts.forEach((pt, i) => {
      if (segments.length === 0 || corners[i]
      || lines[i] || reversions[i] || xExtremas[i] || yExtremas[i]) {
        segments.push({
          cornerIn: corners[i], isLine: lines[i], reversion: reversions[i],
          xExtrema: xExtremas[i], yExtrema: yExtremas[i],
          // tangents: [ tangents[i], tangents[i] ],
          indiceRange: [ i, i ]
        })
      } else {
        const lastSegment = segments[segments.length-1]
        // If a line segment ends, start a new
        if (lastSegment.isLine 
        && lastSegment.indiceRange[1] - lastSegment.indiceRange[0] > 0) {
          segments.push({
            cornerIn: corners[i], isLine: lines[i],
            xExtrema: xExtremas[i], yExtrema: yExtremas[i],
            // tangents: [ tangents[i], tangents[i] ],
            indiceRange: [ i, i ]
          })
        } else {
          lastSegment.indiceRange[1] = i
          // lastSegment.tangents[1] = tangents[i]
        }
      }
    })
    if (segments.length !== 0) {
      const first = segments[0], last = segments[segments.length-1]
      if (!first.cornerIn && !first.isLine && !first.reversion
      && !first.xExtrema && !first.yExtrema) {
        first.indiceRange[0] = last.indiceRange[0]
        // first.tangents[0] = last.tangents[0]
        first.cornerIn = last.cornerIn
        first.isLine = last.isLine
        first.reversion = last.reversion
        first.xExtrema = last.xExtrema
        first.yExtrema = last.yExtrema
        segments.pop()
      }
    }
    // Changing the indice ranges to points
    segments.forEach(s => {
      let points
      const lower = s.indiceRange[0], upper = s.indiceRange[1]
      if (lower > upper) {
        points = [ ...pts.slice(lower), ...pts.slice(0, upper+2) ]
      } else {
        points = pts.slice(lower, upper+2)
        if (upper >= pts.length - 1) points.push(pts[0])
      }
      s.pts = points
      delete s.indiceRange
    })
    /** Result array
     * @type { { cornerIn: boolean, isLine: boolean, reversion: boolean,
     * xExtrema: boolean, yExtrema: boolean, 
     * pts: [ number, number ][] }[] } */
    return segments
  }
  
  
  /** Fit segments
   * @param { { cornerIn: boolean, isLine: boolean, reversion: boolean, 
   * xExtrema: boolean, yExtrema: boolean, indiceRange: [ number, number ], 
   * pts: [ number, number ][] } } segments */
  function fitSegment(segment, error) {
    const { cornerIn, isLine, xExtrema, yExtrema, tangents, pts } = segment
    /** @type { 'none' | 'tangent' | 'y' | 'x' } */
    let smoothFlag = 'none'
    if (!cornerIn) {
      smoothFlag = 'tangent'
      if (xExtrema) smoothFlag = 'y'
      if (yExtrema) smoothFlag = 'x'
    }
    /** @type { Array<Array<Array<Number>>> } */
    const bezierSegments = isLine? 
      JSON.parse(JSON.stringify([ [ pts[0], ...pts, pts[1] ] ])):
      fitCurve(pts, error)
    return {
      line: isLine, smooth: smoothFlag, bezierSegments
    }
  }
  
  function polygon2bezier(polygon, {
    angleLimit = 120,
    shortestLine = 20,
    fitError = 20
  } = {}) {
    const segments = segmentPolygon(polygon, 
      { angleLimit, shortestLine }).map(s => fitSegment(s, fitError))
    // Smoothing
    segments.forEach((s, i) => {
      const prev = segments[i > 0? i-1 : segments.length-1]
      switch (s.smooth) {
        case 'none': return
        case 'x': {
          const y = s.bezierSegments[0][0][1]
          s.bezierSegments[0][1][1] = y
          prev.bezierSegments[prev.bezierSegments.length-1][2][1] = y
          return
        }
        case 'y': {
          const x = s.bezierSegments[0][0][0]
          s.bezierSegments[0][1][0] = x
          prev.bezierSegments[prev.bezierSegments.length-1][2][0] = x
          return
        }
        case 'tangent': {
          const pt = s.bezierSegments[0][0]
          const prevC = prev.bezierSegments[prev.bezierSegments.length-1][2]
          const nextC = s.bezierSegments[0][1]
          const nextVec = [ nextC[0] - pt[0], nextC[1] - pt[1] ]
          const prevVec = [ prevC[0] - pt[0], prevC[1] - pt[1] ]
          const tangentVec = [ nextVec[0] - prevVec[0], nextVec[1] - prevVec[1] ]
          const nextVec1 = projectVector(nextVec, tangentVec)
          const prevVec1 = projectVector(prevVec, tangentVec)
          prevC[0] = pt[0] + prevVec1[0]; prevC[1] = pt[1] + prevVec1[1]
          nextC[0] = pt[0] + nextVec1[0]; nextC[1] = pt[1] + nextVec1[1]
          return
        }
      }
    })
    const result = []
    segments.forEach(s => {
      if (s.line) {
        result.push({
          x: s.bezierSegments[0][0][0], y: s.bezierSegments[0][0][1], on: true
        })
      } else {
        s.bezierSegments.forEach(bs => {
          result.push({ x: bs[0][0], y: bs[0][1], on: true })
          result.push({ x: bs[1][0], y: bs[1][1], on: false })
          result.push({ x: bs[2][0], y: bs[2][1], on: false })
        })
      }
    })
    return result
  }

  module.exports = {
    polygon2bezier,
    traceImage,
    simplifyTrace
  }