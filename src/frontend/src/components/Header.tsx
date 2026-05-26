import { RotatingWord } from "./RotatingWord";

interface HeaderProps {
  variant?: "hero" | "compact";
}

export function Header({ variant = "hero" }: HeaderProps) {
  const isHero = variant === "hero";

  return (
    <header
      className={`enter text-center ${isHero ? "pb-2" : "pt-10 pb-6"}`}
      style={{ "--delay": "0ms" } as React.CSSProperties}
    >
      <div className="flex items-center justify-center gap-2 md:gap-3">
        <img
          src="/casana-logo.png"
          alt=""
          aria-hidden
          className={isHero ? "h-24 md:h-28 w-auto" : "h-14 md:h-16 w-auto"}
        />
        <h1
          className={`font-display font-medium tracking-tight ${
            isHero ? "text-6xl md:text-7xl" : "text-4xl md:text-5xl"
          }`}
        >
          Casana
        </h1>
      </div>
      <div className="mx-auto mt-3 h-px w-10 bg-accent" aria-hidden />
      <p
        className={`enter mx-auto mt-6 font-display leading-snug text-ink ${
          isHero ? "text-3xl md:text-4xl" : "text-xl md:text-2xl"
        }`}
        style={{ "--delay": "140ms" } as React.CSSProperties}
      >
        Te ayudo a buscar tu próximo<RotatingWord />
      </p>
    </header>
  );
}
