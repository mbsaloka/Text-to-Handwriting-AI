import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Slider } from "@/components/ui/slider"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

function ControlsPanel({
  inputText,
  setInputText,
  strokeWidth,
  setStrokeWidth,
  bias,
  setBias,
  style,
  setStyle,
  enableAnimation,
  setEnableAnimation,
  animationSpeed,
  setAnimationSpeed,
  generateHandwriting,
  isGenerating
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Handwriting Generator</CardTitle>
        <CardDescription>Enter text to convert it to handwriting</CardDescription>
      </CardHeader>
      <CardContent className="space-y-6">
        <div className="space-y-2">
          <Label htmlFor="text-input">Text Input</Label>
          <Input
            id="text-input"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            minLength={5}
            maxLength={75}
            required
          />
          {(inputText.length < 5 || inputText.length > 100) && (
            <p className="text-sm text-red-500">
              Text must be in range 5–75 characters.
            </p>
          )}
        </div>

        <div className="space-y-2">
          <Label htmlFor="style-select">Style</Label>
          <Select value={style} onValueChange={setStyle}>
            <SelectTrigger><SelectValue placeholder="Select style" /></SelectTrigger>
            <SelectContent>
              <SelectItem value="0" className="cursor-pointer hover:!bg-background">Random</SelectItem>
              <SelectItem value="1" className="cursor-pointer hover:!bg-background">Style 1</SelectItem>
              <SelectItem value="2" className="cursor-pointer hover:!bg-background">Style 2</SelectItem>
              <SelectItem value="3" className="cursor-pointer hover:!bg-background">Style 3</SelectItem>
              <SelectItem value="4" className="cursor-pointer hover:!bg-background">Style 4</SelectItem>
              <SelectItem value="5" className="cursor-pointer hover:!bg-background">Style 5</SelectItem>
              <SelectItem value="6" className="cursor-pointer hover:!bg-background">Style 6</SelectItem>
              <SelectItem value="7" className="cursor-pointer hover:!bg-background">Style 7</SelectItem>
              <SelectItem value="8" className="cursor-pointer hover:!bg-background">Style 8</SelectItem>
              <SelectItem value="9" className="cursor-pointer hover:!bg-background">Style 9</SelectItem>
              <SelectItem value="10" className="cursor-pointer hover:!bg-background">Style 10</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <Label>Stroke Width: {strokeWidth}</Label>
          <Slider min={1} max={5} step={0.5} value={[strokeWidth]} onValueChange={([v]) => setStrokeWidth(v)} />
        </div>

        <div className="space-y-2">
          <Label>Bias: {bias}</Label>
          <Slider min={0} max={10} step={0.25} value={[bias]} onValueChange={([v]) => setBias(v)} />
        </div>

        <div className="space-y-2">
          <Label>Animation Speed: {animationSpeed}</Label>
          <Slider min={1} max={50} step={1} value={[animationSpeed]} onValueChange={([v]) => setAnimationSpeed(v)} disabled={!enableAnimation} />
        </div>

        <div className="flex items-center space-x-2">
          <Switch checked={enableAnimation} onCheckedChange={setEnableAnimation} />
          <Label>Enable Animation</Label>
        </div>

        <Button onClick={generateHandwriting} disabled={isGenerating || !inputText.trim()} className="w-full">
          {isGenerating ? "Generating..." : "Generate"}
        </Button>
      </CardContent>
    </Card>
  )
}

export default ControlsPanel
