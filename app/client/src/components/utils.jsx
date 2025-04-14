export const convertPointsToSVGPath = (points) => {
  if (points.length === 0) return ""
  let path = "", penDown = false
  points.forEach(([penStatus, x, y], index) => {
    if (penStatus === 1 || index === 0) {
      path += `M${x},${y} `
      penDown = false
    } else if (!penDown) {
      path += `M${x},${y} `
      penDown = true
    } else {
      path += `L${x},${y} `
    }
  })
  return path
}
