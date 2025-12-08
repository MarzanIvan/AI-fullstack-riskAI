// Мобильное меню
$('.contacts__menu-bt').on('click', function(e) {
    $(this).toggleClass("active");
    $('.menu-mobile').toggleClass("active");
    $('body').toggleClass("menu-mobile--opened");
    e.preventDefault();
});

// Фильтр на главной
if (!$('.filter__item.active').length) {
    $('.filter__item[data-filter="all"]').addClass('active');
}

$('.filter button').click(function(e){
    e.preventDefault();

    $(".filter button").removeClass('active');
    $(this).addClass('active');

    var filter = $(this).attr('data-filter');
    if (filter == 'all') {
        $(".stack div").show();
    } else {
        $(".stack__item").hide();
        $(".stack").find("[data-filter-value*='" + filter + "']").show();
    }
});