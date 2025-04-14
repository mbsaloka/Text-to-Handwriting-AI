import { useEffect, useState } from 'react';
import { Link } from "react-router-dom"
import { Link as ScrollLink } from 'react-scroll';
import { useInView } from 'react-intersection-observer';
import { Button } from "@/components/ui/button"
import { ArrowRight } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card";
import { Signature, SlidersVertical, Download } from 'lucide-react';
import DemosCarousel from '@/components/DemosCarousel';

function LandingPage() {
  const [showContent, setShowContent] = useState(true);
  const [ref, inView] = useInView({
    threshold: 0.3,
  });

  const features = [
    { icon: Signature, title: "Realistic Handwriting", description: "Generate authentic-looking handwriting from any text input using deep learning models." },
    { icon: SlidersVertical, title: "Customizable Parameters", description: "Adjust stroke width, animation speed, and style to create the perfect handwritten text." },
    { icon: Download, title: "Export Options", description: "Download your generated handwriting as PNG or SVG files for use in your projects." },
  ];

  useEffect(() => {
    setShowContent(inView);
  }, [inView]);

  return (
    <main className="flex min-h-screen flex-col">
      {/* Hero Section */}
      <section
        className="flex flex-col items-center justify-center min-h-screen text-center space-y-6 px-4 py-20 w-full max-w-7xl mx-auto"
        style={{ transform: 'translateY(-8vh)' }}
      >
        <h1 className="text-4xl font-extrabold tracking-tight lg:text-5xl bg-gradient-to-r from-orange-400 to-red-600 bg-clip-text text-transparent leading-normal h-14">
          Text-to-Handwriting Generation AI
        </h1>
        <p className="text-xl text-muted-foreground max-w-[700px] mx-auto">
          Convert text to realistic handwriting strokes using deep learning models.
        </p>
        <div className="flex justify-center gap-4">
          <Button asChild size="lg" variant="outline">
            <ScrollLink to='about' smooth={true} duration={500} offset={-80}>Learn More</ScrollLink>
          </Button>
          <Button asChild size="lg">
            <Link to="/generator">Try the Demo</Link>
          </Button>
        </div>
      </section>

      {/* About Section */}
      <div ref={ref} className="w-full">
        <section id="about" className={`w-full pt-20 space-y-16 min-h-[70vh] transition-all duration-500 ${showContent ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-10'}`}>
          <div className="text-center space-y-6 max-w-7xl mx-auto px-4">
            <h2 className="text-3xl font-bold tracking-tight lg:text-4xl bg-gradient-to-r from-orange-400 to-red-600 bg-clip-text text-transparent">
              About the Project
            </h2>
            <p className="text-xl text-muted-foreground max-w-[800px] mx-auto">
            This demo showcases a deep learning model that can generate realistic handwriting strokes from text input. The model has been trained on thousands of handwriting samples to learn the nuances of human handwriting.
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 max-w-7xl mx-auto px-4 justify-items-center">
            {features.map((feature, index) => (
              <Card key={index} className={`transition-all duration-500 delay-${index * 100}`}>
                <CardContent className="flex flex-col items-center text-center p-6 space-y-4">
                  <feature.icon className="h-12 w-12 text-primary" />
                  <h3 className="text-lg font-semibold">{feature.title}</h3>
                  <p className="text-sm text-muted-foreground">{feature.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* <div className="max-w-6xl mx-auto">
            <h2 className="text-3xl font-bold text-center mt-24 mb-4 text-foreground">Explore Other Demos</h2>
            <DemosCarousel />
          </div> */}
        </section>
      </div>
    </main>
  )
}

export default LandingPage
