import {useEffect} from "react";

function NotFound() {
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
            <div className="website-wrapper">
                <div className="notfound__wrap">
                    <div className="notfound" title="404">404</div>
                    <div className="notfound__text">Никогда такого не было и вот опять</div>
                    <div className="hero__control">
                        <a href="/" className="button">На главную страницу</a>
                    </div>
                </div>
            </div>
        </>
    )
}

export default NotFound