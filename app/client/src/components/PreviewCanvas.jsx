import { Stage, Layer, Line } from "react-konva"

function PreviewCanvas({ displayedPoints, strokeWidth, stageRef, canvasSize, offsetX = 50, offsetY = 200, scale = 2.5 }) {
  const groupPointsIntoLines = (points) => {
    const lines = []
    let currentLine = []

    points.forEach(([penStatus, x, y], index) => {
      const adjustedX = x * scale + offsetX
      const adjustedY = offsetY - y * scale

      if (penStatus === 1 || index === 0) {
        currentLine.push(adjustedX, adjustedY)
        if (currentLine.length) lines.push({ points: [...currentLine], strokeWidth })
        currentLine = []
      } else {
        currentLine.push(adjustedX, adjustedY)
      }
    })

    if (currentLine.length) lines.push({ points: currentLine, strokeWidth })
    return lines
  }

  const lines = groupPointsIntoLines(displayedPoints)

  return (
    <Stage width={canvasSize.width} height={canvasSize.height} ref={stageRef}>
      <Layer>
        {lines.map((line, index) => (
          <Line
            key={index}
            points={line.points}
            stroke="black"
            strokeWidth={line.strokeWidth}
            lineCap="round"
            lineJoin="round"
          />
        ))}
      </Layer>
    </Stage>
  )
}

export default PreviewCanvas
