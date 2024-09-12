import Projects from "./componetns/projects/projects";
import s from "./page.module.css";

export default function Home() {
  return (
    <div className={s.mainCont}>
      <div className={s.title}>ECG Heartbeat Categorization</div>
      <div className={s.desc}>
        Este trabajo aborda el desarrollo de un modelo de clasificación para
        identificar diversas afecciones cardíacas a partir de lecturas de
        electro- cardiogramas (ECG). Dado que las enfermedades cardiovasculares
        son la principal causa de mortalidad en México, este proyecto se centra
        en mejorar la detección temprana de condiciones como arritmias, con el
        fin de reducir las tasas de mortalidad y mejorar la atención médica.
      </div>
      <Projects />
    </div>
  );
}
