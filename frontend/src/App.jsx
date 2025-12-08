import {useEffect} from "react";
import './App.css'
import SUZUYAlogo from '/src/assets/logo.svg'
import './main.min.css'
import Footer from '/src/Footer.jsx';

import LisSkinsBlack from './assets/tbank whiteblack.svg';
import AvanMarketBlack from "./assets/sber whiteblack.svg";
import TradeItBlack from './assets/tradeitblack.svg';
import YoutubeCircle from './assets/youtubelogocircle.svg';
import HeaderSVG from "./Header SVG.jsx";

function App() {
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
        <div>
            <div className="website-wrapper">
                <header className="header">
                    <div className="container container--fluid">
                        <div className="logotype">
                            <a href="/" className="logotype__link borderless">
                                <div className="logotype__link borderless">
                                    <img src={SUZUYAlogo} width="32" height="32" alt="logotype"/>
                                </div>
                            </a>
                        </div>
                        <nav className="menu">
                            <div className="menu__item">
                                <a href="/contacts/" className="menu__link">
                                    Контакты
                                </a>
                            </div>
                        </nav>
                        <nav className="menu">
                            <div className="menu__item">
                                <a href="/projects/" className="menu__link">
                                    Услуги
                                </a>
                            </div>
                        </nav>

                        <div className="contacts">
                            <a target="_blank" title="Телеграм" className="contacts__round-bt contacts__round-bt--tg"
                               href="https://t.me/dieserpaniker">
                                <svg width="25" height="24" viewBox="0 0 25 24" fill="none"
                                     xmlns="http://www.w3.org/2000/svg">
                                    <path
                                        d="M8.66481 14.2746L15.8952 7.74934C16.2125 7.4677 15.8259 7.33036 15.4047 7.58582L6.48107 13.2156L2.62657 12.0125C1.79416 11.7577 1.78819 11.1857 2.81345 10.7745L17.8336 4.98276C18.5196 4.67131 19.1817 5.14752 18.9198 6.19749L16.3619 18.2514C16.1833 19.108 15.6657 19.3128 14.9486 18.9172L11.0521 16.0383L9.17916 17.8595C9.17326 17.8652 9.16738 17.871 9.16154 17.8767C8.95206 18.0806 8.77875 18.2493 8.39951 18.2493L8.66481 14.2746Z"
                                        fill="black"/>
                                </svg>
                            </a>
                            <a target="_blank" title="YouTube" className="contacts__round-bt contacts__round-bt--wa"
                               href="https://www.youtube.com/@SUZUYATRADE">
                                <img src={YoutubeCircle} width="25" height="24" alt="logotype"/>

                            </a>
                            <a title="Меню" className="contacts__round-bt contacts__menu-bt" href="#">
                                <svg className="ico-burger" width="24" height="24" viewBox="0 0 24 24" fill="none"
                                     xmlns="http://www.w3.org/2000/svg">
                                    <path fillRule="evenodd" clipRule="evenodd"
                                          d="M2 8.5C2 7.94772 2.44772 7.5 3 7.5H21C21.5523 7.5 22 7.94772 22 8.5C22 9.05228 21.5523 9.5 21 9.5H3C2.44772 9.5 2 9.05228 2 8.5ZM2 15.5C2 14.9477 2.44772 14.5 3 14.5H21C21.5523 14.5 22 14.9477 22 15.5C22 16.0523 21.5523 16.5 21 16.5H3C2.44772 16.5 2 16.0523 2 15.5Z"
                                          fill="black"/>
                                </svg>
                                <svg className="ico-close" width="24" height="24" viewBox="0 0 24 24" fill="none"
                                     xmlns="http://www.w3.org/2000/svg">
                                    <path fillRule="evenodd" clipRule="evenodd"
                                          d="M5.29289 5.29289C5.68342 4.90237 6.31658 4.90237 6.70711 5.29289L12 10.5858L17.2929 5.29289C17.6834 4.90237 18.3166 4.90237 18.7071 5.29289C19.0976 5.68342 19.0976 6.31658 18.7071 6.70711L13.4142 12L18.7071 17.2929C19.0976 17.6834 19.0976 18.3166 18.7071 18.7071C18.3166 19.0976 17.6834 19.0976 17.2929 18.7071L12 13.4142L6.70711 18.7071C6.31658 19.0976 5.68342 19.0976 5.29289 18.7071C4.90237 18.3166 4.90237 17.6834 5.29289 17.2929L10.5858 12L5.29289 6.70711C4.90237 6.31658 4.90237 5.68342 5.29289 5.29289Z"
                                          fill="black"/>
                                </svg>
                            </a>
                        </div>
                    </div>
                </header>
                <div className="menu-mobile">
                    <div className="menu-mobile__content">
                        <nav className="menu-mobile__menu">
                            <div className="menu-mobile__item">
                                <a href="/contacts/" className="menu-mobile__link">
                                    Контакты
                                </a>
                            </div>
                        </nav>
                    </div>
                    <div className="menu-mobile__overlay"></div>
                </div>

                <main className="website-workarea js-website-workarea">
                    <section className="section section--boxed section--partners wow animated">
                        <div className="container container--100">
                            <div className="section__header">
                                <div className="title">riskAI&nbsp;— ваш<br/> партнер в онлайн сделках</div>
                                <div className="subtext">Наша&nbsp;цель — сделать ваши сделки продажи и
                                    обмена выгодными и безопасными.<br/>Мы сотрудничаем только с надёжными и
                                    проверенными партнёрамы
                                </div>
                            </div>
                            <div className="section__body">
                                <div className="partners-array">
                                    <div className="partners-array__element">
                                        <div className="partners-element">
                                            <img src={LisSkinsBlack} alt="logotype"/>
                                        </div>
                                    </div>

                                    <div className="partners-array__element">
                                        <div className="partners-element">
                                            <img src={AvanMarketBlack} alt="logotype"/>
                                        </div>
                                    </div>
                                    <div className="partners-array__element">
                                        <div className="partners-element">
                                            <img src={TradeItBlack} alt="logotype"/>
                                        </div>
                                    </div>

                                </div>
                            </div>
                        </div>
                    </section>
                </main>
                <Footer/>
            </div>
            <HeaderSVG />
        </div>
    )
}

export default App
