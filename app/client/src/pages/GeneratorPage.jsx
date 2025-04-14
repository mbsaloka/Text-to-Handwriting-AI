import { useState, useRef, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { ChevronLeft } from "lucide-react"
import { Link } from "react-router-dom"
import { Client } from "@gradio/client";

import PreviewCanvas from "@/components/PreviewCanvas"
import ControlsPanel from "@/components/ControlsPanel"
import HandwritingToolbar from "@/components/HandwritingToolbar"
import { convertPointsToSVGPath } from "@/components/utils"

const fetchHandwritingStrokes = async (text, bias, style) => {
  const backendType = import.meta.env.VITE_BACKEND_TYPE
  if (backendType === "gradio") {
    const API_URL = import.meta.env.VITE_API_URL;
    const client = await Client.connect(API_URL);
    const result = await client.predict("/predict", { text, bias, style_index: style });

    return result.data[1]
  } else if (backendType === "fastapi") {
    const API_URL = import.meta.env.VITE_API_URL;
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text, bias, style })
    })

    if (!response.ok) {
      throw new Error("Failed to fetch handwriting strokes")
    }

    const data = await response.json()

    return data.strokes
  }
}

function GeneratorPage() {
  const [inputText, setInputText] = useState("Give it a try. Have fun!")
  const [strokes, setStrokes] = useState([])
  const [isGenerating, setIsGenerating] = useState(false)
  const [isAnimating, setIsAnimating] = useState(false)
  const [animationSpeed, setAnimationSpeed] = useState(10)
  const [strokeWidth, setStrokeWidth] = useState(2)
  const [bias, setBias] = useState(2.0)
  const [style, setStyle] = useState("0")
  const [enableAnimation, setEnableAnimation] = useState(true)
  const [displayedPoints, setDisplayedPoints] = useState([])
  const [scale, setScale] = useState(2.0)
  const [offsetX, setOffsetX] = useState(50)
  const [offsetY, setOffsetY] = useState(200)
  const [canvasSize, setCanvasSize] = useState({ width: 600, height: 400 })

  const stageRef = useRef(null)
  const animationRef = useRef(null)
  const containerRef = useRef(null)

  // Generate handwriting strokes
  const generateHandwriting = async () => {
    if (!inputText.trim()) return

    setIsGenerating(true)
    try {
      const newStrokes = await fetchHandwritingStrokes(inputText, bias, parseInt(style))
      setStrokes(newStrokes)

      // Reset animation
      setDisplayedPoints([])

      // Compute scale and offsets
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
      for (const [, x, y] of newStrokes) {
        if (x < minX) minX = x
        if (y < minY) minY = y
        if (x > maxX) maxX = x
        if (y > maxY) maxY = y
      }

      const scaleX = (canvasSize.width - 100) / (maxX - minX)
      const scaleY = (canvasSize.height - 100) / (maxY - minY)
      const scale = Math.min(scaleX, scaleY)

      const centerX = (maxX + minX) / 2
      const centerY = (maxY + minY) / 2

      const offsetX = canvasSize.width / 2 - centerX * scale
      const offsetY = canvasSize.height / 2 + centerY * scale

      setScale(scale)
      setOffsetX(offsetX)
      setOffsetY(offsetY)

      // Start animation if enabled
      if (enableAnimation) {
        setIsAnimating(true)
      } else {
        setDisplayedPoints(newStrokes)
      }
    } catch (error) {
      console.error("Error generating handwriting:", error)
    } finally {
      setIsGenerating(false)
    }
  }

  // Handle animation
  useEffect(() => {
    if (isAnimating && strokes.length > 0) {
      let currentIndex = displayedPoints.length

      const animate = () => {
        if (currentIndex < strokes.length) {
          // Add points based on animation speed
          const pointsToAdd = Math.max(1, Math.floor(animationSpeed / 5))
          const newPoints = [...displayedPoints]

          for (let i = 0; i < pointsToAdd; i++) {
            if (currentIndex < strokes.length) {
              newPoints.push(strokes[currentIndex])
              currentIndex++
            }
          }

          setDisplayedPoints(newPoints)
          animationRef.current = requestAnimationFrame(animate)
        } else {
          setIsAnimating(false)
        }
      }

      animationRef.current = requestAnimationFrame(animate)

      return () => {
        if (animationRef.current) {
          cancelAnimationFrame(animationRef.current)
        }
      }
    }
  }, [isAnimating, strokes, displayedPoints, animationSpeed])

  // Toggle animation
  const toggleAnimation = () => {
    if (isAnimating) {
      setIsAnimating(false)
    } else if (strokes.length > 0 && displayedPoints.length < strokes.length) {
      setIsAnimating(true)
    }
  }

  // Reset animation
  const resetAnimation = () => {
    setDisplayedPoints([])
    setIsAnimating(true)
  }

  // Download as PNG
  const downloadAsPNG = () => {
    if (stageRef.current) {
      const uri = stageRef.current.toDataURL()
      const link = document.createElement("a")
      link.download = "handwriting.png"
      link.href = uri
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    }
  }

  // Download as SVG
  const downloadAsSVG = () => {
    if (stageRef.current) {
      // Convert Konva stage to SVG
      const svg = `
        <svg width="${stageRef.current.width()}" height="${stageRef.current.height()}" xmlns="http://www.w3.org/2000/svg">
          <path d="${convertPointsToSVGPath(displayedPoints)}" fill="none" stroke="black" strokeWidth="${strokeWidth}" />
        </svg>
      `

      const blob = new Blob([svg], { type: "image/svg+xml" })
      const url = URL.createObjectURL(blob)
      const link = document.createElement("a")
      link.download = "handwriting.svg"
      link.href = url
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      URL.revokeObjectURL(url)
    }
  }

  // Find container (canvas) size
  useEffect(() => {
    if (containerRef.current) {
      const { width, height } = containerRef.current.getBoundingClientRect()
      setCanvasSize({ width, height })
    }
  }, [])

  // Call generateHandwriting when first loaded
  useEffect(() => {
    generateHandwriting()
  }, [])

  return (
    <div className="min-h-screen bg-background py-8 px-4">
      <div className="max-w-6xl mx-auto">
        <div className="mb-8">
          <Link
            to="/"
            className="text-muted-foreground hover:text-primary flex items-center group transition-colors"
          >
            <ChevronLeft className="w-5 h-5 mr-1 transition-colors group-hover:text-primary" />
            <span className="text-base">Back to Home</span>
          </Link>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Controls */}
          <ControlsPanel
            inputText={inputText}
            setInputText={setInputText}
            strokeWidth={strokeWidth}
            setStrokeWidth={setStrokeWidth}
            bias={bias}
            setBias={setBias}
            style={style}
            setStyle={setStyle}
            enableAnimation={enableAnimation}
            setEnableAnimation={setEnableAnimation}
            animationSpeed={animationSpeed}
            setAnimationSpeed={setAnimationSpeed}
            generateHandwriting={generateHandwriting}
            isGenerating={isGenerating}
          />

          {/* Canvas */}
          <div className="lg:col-span-2">
            <Card className="h-full">
              <CardHeader className="flex flex-row items-center justify-between">
                <div>
                  <CardTitle>Handwriting Preview</CardTitle>
                  <CardDescription>
                    {displayedPoints.length > 0
                      ? `Showing ${displayedPoints.length} of ${strokes.length} points`
                      : "Generate handwriting to see preview"}
                  </CardDescription>
                </div>
                  <HandwritingToolbar
                    displayedPoints={displayedPoints}
                    strokes={strokes}
                    isAnimating={isAnimating}
                    enableAnimation={enableAnimation}
                    toggleAnimation={toggleAnimation}
                    resetAnimation={resetAnimation}
                    downloadAsPNG={downloadAsPNG}
                    downloadAsSVG={downloadAsSVG}
                  />
              </CardHeader>
              <CardContent>
                <div className="bg-white rounded-lg shadow-inner h-[400px] overflow-hidden">
                  <PreviewCanvas
                    displayedPoints={displayedPoints}
                    strokeWidth={strokeWidth}
                    stageRef={stageRef}
                    canvasSize={canvasSize}
                    offsetX={offsetX}
                    offsetY={offsetY}
                    scale={scale}
                  />
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}

export default GeneratorPage