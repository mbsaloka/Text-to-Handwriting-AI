"use client"

import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { ChevronLeft, ChevronRight } from "lucide-react"

function DemosCarousel() {
  const demoProjects = [
    {
      id: 1,
      title: "Text Summarization",
      description: "AI-powered text summarization tool",
      image: "/placeholder.svg?height=200&width=300",
    },
    {
      id: 2,
      title: "Image Generation",
      description: "Create images from text descriptions",
      image: "/placeholder.svg?height=200&width=300",
    },
    {
      id: 3,
      title: "Speech Recognition",
      description: "Convert spoken language to text",
      image: "/placeholder.svg?height=200&width=300",
    },
  ]

  const [currentIndex, setCurrentIndex] = useState(0)
  const intervalRef = useRef(null)

  // Reset the auto-slide timer when manually navigating
  const resetInterval = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
    }

    intervalRef.current = setInterval(() => {
      setCurrentIndex((prevIndex) => (prevIndex + 1) % demoProjects.length)
    }, 4000)
  }

  // Auto-slide every 4 seconds
  useEffect(() => {
    resetInterval()
    return () => clearInterval(intervalRef.current)
  }, [])

  const goToPrevious = () => {
    setCurrentIndex((prevIndex) => (prevIndex === 0 ? demoProjects.length - 1 : prevIndex - 1))
    resetInterval()
  }

  const goToNext = () => {
    setCurrentIndex((prevIndex) => (prevIndex + 1) % demoProjects.length)
    resetInterval()
  }

  const goToSlide = (index) => {
    setCurrentIndex(index)
    resetInterval()
  }

  // Function to get the correct index for the carousel items
  const getItemIndex = (index) => {
    if (index < 0) {
      return demoProjects.length + index
    } else if (index >= demoProjects.length) {
      return index - demoProjects.length
    }
    return index
  }

  return (
    <div className="py-8 max-w-4xl mx-auto">
      {/* Carousel with previews */}
      <div className="relative overflow-hidden">
        <div className="flex items-center justify-center">
          <div className="relative w-full h-[350px] overflow-visible">
            {/* Previous card (preview) */}
            <div
              className="absolute top-1/2 left-0 w-[30%] -translate-y-1/2 transform -translate-x-1/4 opacity-50 cursor-pointer transition-all duration-500 z-10"
              onClick={goToPrevious}
            >
              <div className="bg-card rounded-lg overflow-hidden shadow-md">
                <img
                  src={demoProjects[getItemIndex(currentIndex - 1)].image || "/placeholder.svg"}
                  alt={demoProjects[getItemIndex(currentIndex - 1)].title}
                  className="w-full h-32 object-cover"
                />
                <div className="p-3">
                  <h3 className="text-sm font-semibold truncate">
                    {demoProjects[getItemIndex(currentIndex - 1)].title}
                  </h3>
                </div>
              </div>
            </div>

            {/* Current card */}
            <div className="absolute top-1/2 left-1/2 w-[50%] -translate-x-1/2 -translate-y-1/2 transition-all duration-500 z-20 scale-110">
              <div className="bg-card rounded-lg overflow-hidden shadow-xl">
                <img
                  src={demoProjects[currentIndex].image || "/placeholder.svg"}
                  alt={demoProjects[currentIndex].title}
                  className="w-full h-40 object-cover"
                />
                <div className="p-4">
                  <h3 className="text-lg font-semibold mb-1 text-card-foreground">
                    {demoProjects[currentIndex].title}
                  </h3>
                  <p className="text-muted-foreground text-sm">{demoProjects[currentIndex].description}</p>
                  <Button variant="link" className="mt-2 p-0 text-primary">
                    View Demo
                  </Button>
                </div>
              </div>
            </div>

            {/* Next card (preview) */}
            <div
              className="absolute top-1/2 right-0 w-[30%] -translate-y-1/2 transform translate-x-1/4 opacity-50 cursor-pointer transition-all duration-500 z-10"
              onClick={goToNext}
            >
              <div className="bg-card rounded-lg overflow-hidden shadow-md">
                <img
                  src={demoProjects[getItemIndex(currentIndex + 1)].image || "/placeholder.svg"}
                  alt={demoProjects[getItemIndex(currentIndex + 1)].title}
                  className="w-full h-32 object-cover"
                />
                <div className="p-3">
                  <h3 className="text-sm font-semibold truncate">
                    {demoProjects[getItemIndex(currentIndex + 1)].title}
                  </h3>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Indicators */}
      <div className="flex justify-center mt-6 gap-2">
        <button onClick={goToPrevious} aria-label="Previous slide">
          <ChevronLeft className="w-6 h-6 text-muted-foreground hover:text-primary transition-colors" />
        </button>
        {demoProjects.map((_, i) => (
          <button
            key={i}
            onClick={() => goToSlide(i)}
            className="focus:outline-none focus-visible:ring-2 focus-visible:ring-primary rounded-full"
            aria-label={`Go to slide ${i + 1}`}
            aria-current={currentIndex === i ? "true" : "false"}
          >
            <span
              className={`block h-2 rounded-full transition-all duration-300 ${
                currentIndex === i ? "w-6 bg-primary" : "w-2 bg-muted-foreground hover:bg-primary/50"
              }`}
            />
          </button>
        ))}
        <button onClick={goToNext} aria-label="Next slide">
          <ChevronRight className="w-6 h-6 text-muted-foreground hover:text-primary transition-colors" />
        </button>
      </div>
    </div>
  )
}

export default DemosCarousel
