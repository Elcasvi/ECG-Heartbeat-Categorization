"use client";
import { useEffect, useState } from "react";
import { Line } from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import data from "./componetns/datos/data.json"; // Adjust the path as necessary
import s from "./page.module.css";
import Projects from "./componetns/projects/projects";

interface ChartData {
  labels: number[];
  datasets: {
    label: string;
    data: number[];
    borderColor: string;
    backgroundColor: string;
    borderWidth: number;
  }[];
}

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

export default function Home() {
  const [chartData, setChartData] = useState<ChartData | null>(null);

  useEffect(() => {
    // Extract the first array from the data
    const firstArray = data.data[0];

    // Prepare chart data
    const chartData: ChartData = {
      labels: firstArray.map((_, index) => index + 1), // X-axis labels as index + 1
      datasets: [
        {
          label: "ECG Data",
          data: firstArray,
          borderColor: "rgba(75, 192, 192, 1)",
          backgroundColor: "rgba(75, 192, 192, 0.2)",
          borderWidth: 1,
        },
      ],
    };

    setChartData(chartData);
  }, []);

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
      {chartData && (
        <div>
          <h2>ECG Data Graph</h2>
          <Line data={chartData} />
        </div>
      )}
    </div>
  );
}
