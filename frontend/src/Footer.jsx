import {useEffect} from "react";
import './Footer.css'
import CLlogo from "./assets/whitelogo.svg";
import LisLogo from "./assets/lis-skins.svg";
import AvanMarket from "./assets/avanmarket.svg";
function Footer() {
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
            <footer aria-label="primary" className="StaticFooter-footerContainer-jf4">
                <div className="StaticFooter-footerContent-OnA">

                    <div className="StaticFooter-mainFooterContainer-Bqa">
                        <div className="StaticFooter-behanceLinkContainer-qBT">
                            <a href="/" className="hidden-mobile logotype__link borderless">
                                <img src={CLlogo} width="32" height="32" alt="logotype"/>
                                <br/>
                                <span className="whitespan">
                                    © 2025. riskAI.<br/>Наша цель <br/> сделать ваши сделки <br/> выгодными и надёжными
                                </span>
                            </a>
                            <br/>
                        </div>
                        <div className="StaticFooter-sectionContainer-Kyo">
                            <div className="StaticFooter-footerSection-E69">
                                <div className="StaticFooter-footerSectionHeader-xFg">riskAI</div>
                                <div>
                                    <ul>
                                        <li className="StaticFooter-footerSelectorItems-I6_"><a
                                            className="StaticFooter-footerSelectorLinks-HKF FooterLinks-link-SQa"
                                            href="/"><p
                                            className="FooterLinks-text-3wz">Риски в кредитовании</p></a></li>

                                        <li className="StaticFooter-footerSelectorItems-I6_"><a
                                            className="StaticFooter-footerSelectorLinks-HKF FooterLinks-link-SQa"
                                            href="/"><p
                                            className="FooterLinks-text-3wz">Риски в стаховании</p></a></li>

                                        <li className="StaticFooter-footerSelectorItems-I6_"><a
                                            className="StaticFooter-footerSelectorLinks-HKF FooterLinks-link-SQa"
                                            href="/"><p
                                            className="FooterLinks-text-3wz">Риски в инвестициях</p></a></li>

                                    </ul>
                                </div>
                            </div>
                            <div className="StaticFooter-footerSection-E69">
                                <div className="StaticFooter-footerSectionHeader-xFg">Social</div>
                                <div>
                                    <ul>
                                    <li className="StaticFooter-footerSelectorItems-I6_"><a
                                            className="StaticFooter-footerSelectorLinks-HKF FooterLinks-link-SQa"
                                            href="https://www.youtube.com/@SUZUYATRADE" target="_blank" tabIndex="-1">
                                            <p className="FooterLinks-text-3wz"><span
                                                className="FooterLinks-iconContainer-T6h StaticFooter-iconClass-JdV">
                                                </span>Youtube
                                            </p>
                                        </a></li>
                                        <li className="StaticFooter-footerSelectorItems-I6_"><a
                                            className="StaticFooter-footerSelectorLinks-HKF FooterLinks-link-SQa"
                                            href="https://rutube.ru/channel/56297444/" target="_blank" tabIndex="-1">
                                            <p className="FooterLinks-text-3wz"><span
                                                className="FooterLinks-iconContainer-T6h StaticFooter-iconClass-JdV">
                                                </span>Rutube
                                            </p>
                                        </a></li>
                                        <li className="StaticFooter-footerSelectorItems-I6_"><a
                                            className="StaticFooter-footerSelectorLinks-HKF FooterLinks-link-SQa"
                                            href="https://vk.com/suzuyatrade" target="_blank" tabIndex="-1">
                                            <p className="FooterLinks-text-3wz"><span
                                                className="FooterLinks-iconContainer-T6h StaticFooter-iconClass-JdV">
                                                </span>VK video
                                            </p>
                                        </a></li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </footer>
        </>
    )
}

export default Footer