"use client";
import s from "./projects.module.css";
import React, { useState, useEffect, useRef } from "react";

import { Swiper, SwiperSlide } from "swiper/react";
import "swiper/css";

export default function Projects() {
  const projects = [
    {
      name: "Ritmo sinusal normal",
      description:
        "El ritmo sinusal normal es un patrón de latido cardíaco regular y coordinado que se origina en el nódulo sinusal, situado en la aurícula derecha del corazón. Este ritmo es caracterizado por una frecuencia cardíaca entre 60 y 100 latidos por minuto en reposo y una secuencia regular de ondas P, complejos QRS y ondas T en un electrocardiograma (ECG), indicando una adecuada función del sistema de conducción cardíaco.",
      img: "/normal.jpeg",
    },
    {
      name: "Arritmias supraventriculares",
      description:
        "Las arritmias supraventriculares son un grupo de trastornos del ritmo cardíaco que se originan en las estructuras localizadas por encima de los ventrículos, principalmente en las aurículas o en el nódulo auriculoventricular. Estas arritmias pueden incluir fibrilación auricular, aleteo auricular y taquicardia supraventricular paroxística. Se caracterizan por una frecuencia cardíaca anormalmente rápida y, a menudo, una irregularidad en el ritmo del ECG.",
      img: "/supra.jpeg",
    },
    {
      name: "Latidos cardíacos desconocidos / no clasificados",
      description:
        "Los latidos cardíacos desconocidos o no clasificados se refieren a patrones en el ECG que no se ajustan claramente a las categorías tradicionales de arritmias cardíacas conocidas. Estos patrones pueden ser atípicos o poco frecuentes y, por lo tanto, no se pueden clasificar con seguridad en una categoría específica, lo que requiere un análisis más profundo o seguimiento para determinar su origen y significado clínico.",
      img: "/unknwn.jpeg",
    },
    {
      name: "Arritmias ventriculares",
      description:
        "Las arritmias ventriculares son trastornos del ritmo cardíaco que se originan en los ventrículos del corazón. Estas arritmias pueden incluir taquicardia ventricular y fibrilación ventricular, y se caracterizan por una actividad eléctrica desorganizada o anormal en los ventrículos, lo que puede resultar en una frecuencia cardíaca rápida, ineficaz y potencialmente peligrosa. Las arritmias ventriculares pueden comprometer el flujo sanguíneo y requerir intervención médica urgente.",
      img: "/ventri.jpeg",
    },
    {
      name: "Latidos de fusión",
      description:
        "Los latidos de fusión son una mezcla de características de ritmos cardíacos normales y anormales. Se producen cuando un latido del corazón incorpora elementos tanto del ritmo sinusal normal como de un ritmo anormal, como en el caso de una arritmia que interfiere con el ritmo normal. En el ECG, estos latidos pueden presentar características intermedias que reflejan la superposición de ambos tipos de ritmos, lo que puede complicar la interpretación del patrón cardíaco.",
      img: "/fusion.jpeg",
    },
  ];

  return (
    <main className={s.mainCont}>
      <Swiper
        className={s.slider}
        direction={"horizontal"}
        slidesPerView={"auto"}
        centeredSlides
        loop
        initialSlide={2}
        autoHeight={false}
      >
        {projects.map((project, index) => (
          <SwiperSlide className={s.slide} key={index}>
            <div className={`${s.projectCont} ${s.mobile}`}>
              <div className={s.imgCont}>
                <img draggable="false" className={s.img} src={project.img} />
              </div>
              <div className={s.project}>
                <div className={s.projectInner}>
                  <h2 className={s.title}>{project.name}</h2>
                  <p className={s.desc}>{project.description}</p>
                </div>
              </div>
            </div>
          </SwiperSlide>
        ))}
      </Swiper>
    </main>
  );
}
