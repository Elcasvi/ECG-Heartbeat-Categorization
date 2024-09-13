"use client";
import { useEffect, useState } from "react";
import { VictoryChart, VictoryLine, VictoryAxis, VictoryTheme } from "victory";
import data from "./components/datos/data.json";
import s from "./page.module.css";
import Projects from "./components/projects/projects";

export default function Home() {
  const categories = [
    {
      name: "Normal",
    },
    {
      name: "Supraventriculares",
    },
    {
      name: "Desconocidos",
    },
    {
      name: "Ventriculares",
    },
    {
      name: "Fusión",
    },
  ];

  const categoryMapping: { [key: string]: string } = {
    S: categories[1].name, // Supraventriculares
    Q: categories[2].name, // Desconocidos
    V: categories[3].name, // Ventriculares
    F: categories[4].name, // Fusión
  };

  const [chartData, setChartData] = useState<{ x: number; y: number }[] | null>(
    null
  );
  const [selectedCategory, selectCategory] = useState(0);
  const [loading, setLoading] = useState(false);
  const [responseData, setResponseData] = useState(null);

  const apiEndpoint =
    "https://ecg-classifier-app-ctang6ahckcdb8gd.eastus-01.azurewebsites.net/predict_illness";

  const formatData = (index: number) => {
    const formattedData = data.data[index].map((value, index) => ({
      x: index + 1,
      y: value,
    }));
    return formattedData;
  };

  useEffect(() => {
    setChartData(formatData(0));
  }, []);

  const handleCategoryClick = (index: number) => {
    setChartData(formatData(index));
    selectCategory(index);
  };

  const handleApiCall = async () => {
    setLoading(true);
    try {
      const response = await fetch(apiEndpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ data: data.data[selectedCategory] }),
      });
      if (response.ok) {
        const result = await response.json();
        setResponseData(result);
      } else {
        console.error("API request failed:", response.statusText);
      }
    } catch (error) {
      console.error("Error during API request:", error);
    }
    setLoading(false);
  };

  return (
    <div className={s.mainCont}>
      <div className={s.section}>
        <div className={s.title}>ECG Heartbeat Categorization</div>
        <div className={s.desc}>
          Este trabajo aborda el desarrollo de un modelo de clasificación para
          identificar diversas afecciones cardíacas a partir de lecturas de
          electrocardiogramas (ECG). Dado que las enfermedades cardiovasculares
          son la principal causa de mortalidad en México, este proyecto se
          centra en mejorar la detección temprana de condiciones como arritmias,
          con el fin de reducir las tasas de mortalidad y mejorar la atención
          médica.
        </div>
        <Projects />
      </div>

      <div className={s.section}>
        <div className={s.subtitle}>Selecciona una muestra para analizaar</div>
        <div className={s.categories}>
          {categories
            .filter((category) => category.name.toLowerCase() !== "normal")
            .map((category, index) => (
              <div
                key={index}
                onClick={() => handleCategoryClick(index + 1)}
                className={
                  selectedCategory != index + 1
                    ? s.category
                    : s.categorySelected
                }
              >
                {category.name}
              </div>
            ))}
        </div>
        {chartData && (
          <div>
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
                animate={{
                  duration: 1000, // Animation duration in milliseconds
                  onLoad: { duration: 1000 }, // Initial loading animation
                  easing: "linear", // Type of easing function (e.g., "linear", "bounce", etc.)
                }}
              />
            </VictoryChart>
          </div>
        )}
        <button
          onClick={handleApiCall}
          disabled={loading}
          className={s.category}
        >
          {loading ? "Loading..." : "Analizar datos con AdaBoost"}
        </button>
        {responseData && (
          <div className={s.resultCont}>
            <div>{"Resultado de la API "}</div>
            <div>
              {" "}
              {responseData}
              {": "}
              {categoryMapping[responseData]}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
