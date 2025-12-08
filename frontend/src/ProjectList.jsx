import {useEffect} from "react";

function ProjectList () {
    // eslint-disable-next-line react-hooks/rules-of-hooks
    useEffect(() => {
        const script = document.createElement('script');
        script.src = '../src/min_main.js';
        script.async = true;
        const mobilescript = document.createElement('script');
        mobilescript.src = '../src/custom.js';
        mobilescript.async = true;
        document.body.appendChild(mobilescript);
        document.body.appendChild(script);
        return () => {
            document.body.removeChild(script);
            document.body.removeChild(mobilescript);
        };
    }, []);
    return (
        <>
            <div className="portfolio-array">
                <div className="portfolio-array__element">
                    <div className="portfolio-element">
                        <a href="/credit" className="portfolio-element__link borderless"
                           title="Риск-management в кредитовании с riskAI">
                            <div className="portfolio-element__image image-loading">
                                <img className="lazyload" src="/img/item1.jpg"
                                     srcSet=""
                                     alt="Кредитование" loading="lazy"/>
                            </div>
                            <div className="portfolio-element__details">
                                <div className="portfolio-element__text">Риск-management в кредитовании с riskAI
                                </div>
                            </div>
                        </a>
                    </div>
                </div>
                <div className="portfolio-array__element">
                    <div className="portfolio-element">
                        <a href="/insurance" className="portfolio-element__link borderless"
                           title="Риск-management в страховании с riskAI">
                            <div className="portfolio-element__image image-loading">
                                <img className="lazyload" src="/img/item2.jpg"
                                     srcSet=""
                                     alt="Страхование. " loading="lazy"/>
                            </div>
                            <div className="portfolio-element__details">
                                <div className="portfolio-element__text">Риск-management в страховании с riskAI
                                </div>
                            </div>
                        </a>
                    </div>
                </div>
                <div className="portfolio-array__element">
                    <div className="portfolio-element">
                        <a href="/investment" className="portfolio-element__link borderless"
                           title="Риск-management в инвестировании с riskAI">
                            <div className="portfolio-element__image image-loading">
                                <img className="lazyload" src="/img/item3.jpg"
                                     srcSet=""
                                     alt="Инвестиции. " loading="lazy"/>
                            </div>
                            <div className="portfolio-element__details">
                                <div className="portfolio-element__text">Риск-management в инвестировании с riskAI
                                </div>
                            </div>
                        </a>
                    </div>
                </div>
            </div>
        </>
    )
}

export default ProjectList