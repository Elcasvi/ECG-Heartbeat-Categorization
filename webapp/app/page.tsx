"use client";
import { useEffect, useState } from "react";
import { VictoryChart, VictoryLine, VictoryAxis, VictoryTheme } from "victory";
import data from "./components/datos/data.json";
import s from "./page.module.css";
import Projects from "./components/projects/projects";

export default function Home() {
  const [chartData, setChartData] = useState<{ x: number; y: number }[] | null>(
    null
  );

  useEffect(() => {
    // Extract the first array from the data
    const firstArray = data.data[0];

    // Prepare chart data
    const formattedData = firstArray.map((value, index) => ({
      x: index + 1, // X-axis as index + 1
      y: value,
    }));

    setChartData(formattedData);
  }, []);

  return (
    <div className={s.mainCont}>
      <div className={s.title}>ECG Heartbeat Categorization</div>
      <div className={s.desc}>
        Este trabajo aborda el desarrollo de un modelo de clasificación para
        identificar diversas afecciones cardíacas a partir de lecturas de
        electrocardiogramas (ECG). Dado que las enfermedades cardiovasculares
        son la principal causa de mortalidad en México, este proyecto se centra
        en mejorar la detección temprana de condiciones como arritmias, con el
        fin de reducir las tasas de mortalidad y mejorar la atención médica.
      </div>
      <Projects />
      {chartData && (
        <div>
          <h2>ECG Data Graph</h2>
          <VictoryChart width={700} theme={VictoryTheme.material}>
            <VictoryAxis
              tickFormat={(x: { toString: () => any }) => x.toString()}
            />
            <VictoryAxis
              dependentAxis
              tickFormat={(y: number) => y.toFixed(2)}
            />
            <VictoryLine
              data={chartData}
              x="x"
              y="y"
              style={{
                data: { stroke: "rgba(255, 0, 0, 1)" },
              }}
            />
          </VictoryChart>
        </div>
      )}
    </div>
  );
}
