/** Signal Observatory ambient layer: restrained teal/amber motion, no inputs or network calls. */
import { useEffect, useRef } from "react";

type Particle = { x: number; y: number; driftX: number; driftY: number; radius: number; tone: "teal" | "amber" };

export function AmbientField() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d", { alpha: true });
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    if (!context || reducedMotion.matches) return;

    let width = 0;
    let height = 0;
    let frame = 0;
    let active = !document.hidden;
    let particles: Particle[] = [];
    let lastDraw = 0;

    const createParticle = (index: number): Particle => ({
      x: Math.random() * width,
      y: Math.random() * height,
      driftX: (Math.random() - 0.5) * 0.085,
      driftY: (Math.random() - 0.5) * 0.065,
      radius: 0.65 + Math.random() * 1.45,
      tone: index % 7 === 0 ? "amber" : "teal",
    });

    const resize = () => {
      const ratio = Math.min(window.devicePixelRatio || 1, 1.5);
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = Math.floor(width * ratio);
      canvas.height = Math.floor(height * ratio);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      const count = width < 700 ? 9 : 16;
      particles = Array.from({ length: count }, (_, index) => createParticle(index));
    };

    const draw = (time: number) => {
      context.clearRect(0, 0, width, height);
      const drift = time * 0.00011;
      const wash = context.createRadialGradient(width * (0.68 + Math.sin(drift) * 0.07), height * (0.23 + Math.cos(drift * 0.7) * 0.05), 0, width * 0.68, height * 0.23, Math.max(width, height) * 0.66);
      wash.addColorStop(0, "rgba(49, 211, 198, 0.1)");
      wash.addColorStop(0.42, "rgba(49, 211, 198, 0.025)");
      wash.addColorStop(1, "rgba(49, 211, 198, 0)");
      context.fillStyle = wash;
      context.fillRect(0, 0, width, height);

      for (let lane = 0; lane < 3; lane += 1) {
        context.strokeStyle = lane === 1 ? "rgba(245, 185, 66, 0.045)" : "rgba(49, 211, 198, 0.065)";
        context.lineWidth = 0.7;
        context.beginPath();
        for (let x = -40; x <= width + 40; x += 34) {
          const y = height * (0.19 + lane * 0.29) + Math.sin(x * 0.0034 + drift * (lane + 1.2)) * (15 + lane * 5);
          if (x < 0) context.moveTo(x, y);
          else context.lineTo(x, y);
        }
        context.stroke();
      }

      for (const particle of particles) {
        particle.x += particle.driftX;
        particle.y += particle.driftY;
        if (particle.x < -12 || particle.x > width + 12) particle.driftX *= -1;
        if (particle.y < -12 || particle.y > height + 12) particle.driftY *= -1;
      }

      context.lineWidth = 0.55;
      for (let left = 0; left < particles.length; left += 1) {
        for (let right = left + 1; right < particles.length; right += 1) {
          const a = particles[left];
          const b = particles[right];
          const distance = Math.hypot(a.x - b.x, a.y - b.y);
          if (distance > 150) continue;
          context.strokeStyle = `rgba(49, 211, 198, ${0.028 * (1 - distance / 150)})`;
          context.beginPath();
          context.moveTo(a.x, a.y);
          context.lineTo(b.x, b.y);
          context.stroke();
        }
      }

      for (const particle of particles) {
        const color = particle.tone === "teal" ? "49, 211, 198" : "245, 185, 66";
        const halo = context.createRadialGradient(particle.x, particle.y, 0, particle.x, particle.y, particle.radius * 7);
        halo.addColorStop(0, `rgba(${color}, 0.15)`);
        halo.addColorStop(1, `rgba(${color}, 0)`);
        context.fillStyle = halo;
        context.beginPath();
        context.arc(particle.x, particle.y, particle.radius * 7, 0, Math.PI * 2);
        context.fill();
        context.fillStyle = `rgba(${color}, 0.34)`;
        context.beginPath();
        context.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2);
        context.fill();
      }
    };

    const animate = (time: number) => {
      if (!active) return;
      if (time - lastDraw >= 33) {
        draw(time);
        lastDraw = time;
      }
      frame = window.requestAnimationFrame(animate);
    };

    const handleVisibility = () => {
      active = !document.hidden && !reducedMotion.matches;
      if (active) frame = window.requestAnimationFrame(animate);
      else window.cancelAnimationFrame(frame);
    };

    const handleMotionPreference = () => {
      active = !document.hidden && !reducedMotion.matches;
      if (!active) {
        context.clearRect(0, 0, width, height);
        window.cancelAnimationFrame(frame);
      } else frame = window.requestAnimationFrame(animate);
    };

    resize();
    frame = window.requestAnimationFrame(animate);
    window.addEventListener("resize", resize, { passive: true });
    document.addEventListener("visibilitychange", handleVisibility);
    reducedMotion.addEventListener("change", handleMotionPreference);
    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener("resize", resize);
      document.removeEventListener("visibilitychange", handleVisibility);
      reducedMotion.removeEventListener("change", handleMotionPreference);
    };
  }, []);

  return <canvas ref={canvasRef} className="ambient-field" aria-hidden="true" />;
}
