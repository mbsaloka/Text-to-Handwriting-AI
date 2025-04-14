import { Outlet, Link } from 'react-router-dom';
import { Button } from "@/components/ui/button";
import { useTheme } from "./ThemeProvider";
import { Pencil, Moon, Sun } from 'lucide-react';

function Layout() {
  const { theme, setTheme } = useTheme();
  const toggleTheme = () => setTheme(theme === "dark" ? "light" : "dark");

  return (
    <div className="flex flex-col min-h-screen">
      <nav className={`fixed w-full z-20 top-0 start-0 border-b bg-muted text-primary`}>
        <div className="max-w-screen-2xl flex items-center justify-between mx-auto p-4">
          <Link to="/" className="flex items-center space-x-3 rtl:space-x-reverse">
            <Pencil size={28} className="text-orange-400" />
            <span className="sm:text-2xl self-center text-xl font-semibold whitespace-nowrap bg-gradient-to-r from-orange-400 to-red-600 bg-clip-text text-transparent">Handwriting AI</span>
          </Link>

          <div className="flex items-center space-x-3 md:order-2">
            <Button variant="ghost" size="icon" onClick={toggleTheme} className="hidden md:flex">
              {theme === "dark" ? <Sun className="h-[1.2rem] w-[1.2rem]" /> : <Moon className="h-[1.2rem] w-[1.2rem]" />}
            </Button>
          </div>
        </div>
      </nav>

      <main className="flex-grow w-full transform translate-y-16">
        <div className="container mx-auto px-4 py-8">
          <Outlet />
        </div>
      </main>

      <footer className="bg-muted w-full mt-12">
        <div className="container mx-auto px-4 py-6 text-center">
          <p>&copy; 2025 Handwriting AI Demo App. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}

export default Layout;
