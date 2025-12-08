import {useEffect} from "react";

function indexProjects() {
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
                        <a href="/" className="portfolio-element__link borderless"
                           title="Виджеты группы вашего бизнеса">
                            <div className="portfolio-element__image image-loading">
                                <img className="lazyload" src="/img/item1.jpg"
                                     srcSet=""
                                     alt="Стильный дизайн. " loading="lazy"/>
                            </div>
                            <div className="portfolio-element__details">
                                <div className="portfolio-element__text">Дизайн виджета для
                                    интернет-магазина
                                </div>
                            </div>
                        </a>
                    </div>
                </div>

                <div className="portfolio-array__element">
                    <div className="portfolio-element">
                        <a href="/" className="portfolio-element__link borderless"
                           title="Разработка логотипа">
                            <div className="portfolio-element__image image-loading">
                                <img className="lazyload" src="/img/item2.jpg"
                                     srcSet=""
                                     alt="Стильный дизайн. " loading="lazy"/>
                            </div>
                            <div className="portfolio-element__details">
                                <div className="portfolio-element__text">Разработка логотипа
                                </div>
                            </div>
                        </a>
                    </div>
                </div>

                <div className="portfolio-array__element">
                    <div className="portfolio-element">
                        <a href="/" className="portfolio-element__link borderless"
                           title="Плакаты">
                            <div className="portfolio-element__image image-loading">
                                <img className="lazyload" src="/img/item3.jpg"
                                     srcSet=""
                                     alt="Стильный дизайн. " loading="lazy"/>
                            </div>
                            <div className="portfolio-element__details">
                                <div className="portfolio-element__text">Оформление плакатов для
                                    бизнеса
                                </div>
                            </div>
                        </a>
                    </div>
                </div>

                <div className="portfolio-array__element">
                    <div className="portfolio-element">
                        <a href="/" className="portfolio-element__link borderless"
                           title="Посты для ввдения социальных медия">
                            <div className="portfolio-element__image image-loading">
                                <img className="lazyload" src="/img/item4.jpg"
                                     srcSet=""
                                     alt="Стильный дизайн. " loading="lazy"/>
                            </div>
                            <div className="portfolio-element__details">
                                <div className="portfolio-element__text">Оформление постов
                                </div>
                            </div>
                        </a>
                    </div>
                </div>

                <div className="portfolio-array__element">
                    <div className="portfolio-element">
                        <a href="/" className="portfolio-element__link borderless"
                           title="Визитка">
                            <div className="portfolio-element__image image-loading">
                                <img className="lazyload" src="/img/item5.jpg"
                                     srcSet=""
                                     alt="Стильный дизайн. " loading="lazy"/>
                            </div>
                            <div className="portfolio-element__details">
                                <div className="portfolio-element__text">Создание визиток
                                </div>
                            </div>
                        </a>
                    </div>
                </div>

                <div className="portfolio-array__element">
                    <div className="portfolio-element">
                        <a href="/" className="portfolio-element__link borderless"
                           title="Дизайн, который поможет вашему бизнесу">
                            <div className="portfolio-element__image image-loading">
                                <img className="lazyload" src="/img/item6.jpg"
                                     srcSet=""
                                     alt="Стильный дизайн. " loading="lazy"/>
                            </div>
                            <div className="portfolio-element__details">
                                <div className="portfolio-element__text">Запоминающийся дизайн
                                </div>
                            </div>
                        </a>
                    </div>
                </div>
            </div>
        </>
    )
}

export default indexProjects