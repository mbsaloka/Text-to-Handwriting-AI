import { Button } from "@/components/ui/button"
import { Pause, Play, RefreshCw, Download } from "lucide-react"
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from "@/components/ui/dropdown-menu"

function HandwritingToolbar({
  displayedPoints,
  strokes,
  isAnimating,
  enableAnimation,
  toggleAnimation,
  resetAnimation,
  downloadAsPNG,
  downloadAsSVG
}) {
  return (
    <div className="flex space-x-2">
      {displayedPoints.length > 0 && (
        <>
          <Button variant="outline" size="icon" onClick={toggleAnimation} disabled={!enableAnimation || displayedPoints.length >= strokes.length}>
            {isAnimating ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
          </Button>
          <Button variant="outline" size="icon" onClick={resetAnimation} disabled={!enableAnimation || displayedPoints.length === 0}>
            <RefreshCw className="h-4 w-4" />
          </Button>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="icon">
                <Download className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem onClick={downloadAsPNG} className="cursor-pointer hover:!bg-background">Download PNG</DropdownMenuItem>
              <DropdownMenuItem onClick={downloadAsSVG} className="cursor-pointer hover:!bg-background">Download SVG</DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </>
      )}
    </div>
  )
}

export default HandwritingToolbar
