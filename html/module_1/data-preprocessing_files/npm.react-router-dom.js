"use strict";(self.webpackChunkliveVideoAndProject=self.webpackChunkliveVideoAndProject||[]).push([[150],{76:(e,t,n)=>{n.d(t,{VK:()=>f,rU:()=>m,OL:()=>C});var r=n(15137);function o(e,t){return o=Object.setPrototypeOf||function(e,t){return e.__proto__=t,e},o(e,t)}var a=n(67294),i=n(4852);function c(){return c=Object.assign||function(e){for(var t=1;t<arguments.length;t++){var n=arguments[t];for(var r in n)Object.prototype.hasOwnProperty.call(n,r)&&(e[r]=n[r])}return e},c.apply(this,arguments)}function l(e,t){if(null==e)return{};var n,r,o={},a=Object.keys(e);for(r=0;r<a.length;r++)n=a[r],t.indexOf(n)>=0||(o[n]=e[n]);return o}n(45697);var u=n(2177),f=function(e){function t(){for(var t,n=arguments.length,r=new Array(n),o=0;o<n;o++)r[o]=arguments[o];return(t=e.call.apply(e,[this].concat(r))||this).history=(0,i.lX)(t.props),t}return c=e,(n=t).prototype=Object.create(c.prototype),n.prototype.constructor=n,o(n,c),t.prototype.render=function(){return a.createElement(r.F0,{history:this.history,children:this.props.children})},t;var n,c}(a.Component);a.Component;var s=function(e,t){return"function"==typeof e?e(t):e},p=function(e,t){return"string"==typeof e?(0,i.ob)(e,null,null,t):e},v=function(e){return e},y=a.forwardRef;void 0===y&&(y=v);var h=y((function(e,t){var n=e.innerRef,r=e.navigate,o=e.onClick,i=l(e,["innerRef","navigate","onClick"]),u=i.target,f=c({},i,{onClick:function(e){try{o&&o(e)}catch(t){throw e.preventDefault(),t}e.defaultPrevented||0!==e.button||u&&"_self"!==u||function(e){return!!(e.metaKey||e.altKey||e.ctrlKey||e.shiftKey)}(e)||(e.preventDefault(),r())}});return f.ref=v!==y&&t||n,a.createElement("a",f)})),m=y((function(e,t){var n=e.component,o=void 0===n?h:n,f=e.replace,m=e.to,d=e.innerRef,g=l(e,["component","replace","to","innerRef"]);return a.createElement(r.s6.Consumer,null,(function(e){e||(0,u.Z)(!1);var n=e.history,r=p(s(m,e.location),e.location),l=r?n.createHref(r):"",h=c({},g,{href:l,navigate:function(){var t=s(m,e.location),r=(0,i.Ep)(e.location)===(0,i.Ep)(p(t));(f||r?n.replace:n.push)(t)}});return v!==y?h.ref=t||d:h.innerRef=d,a.createElement(o,h)}))})),d=function(e){return e},g=a.forwardRef;void 0===g&&(g=d);var C=g((function(e,t){var n=e["aria-current"],o=void 0===n?"page":n,i=e.activeClassName,f=void 0===i?"active":i,v=e.activeStyle,y=e.className,h=e.exact,C=e.isActive,R=e.location,b=e.sensitive,O=e.strict,j=e.style,k=e.to,w=e.innerRef,E=l(e,["aria-current","activeClassName","activeStyle","className","exact","isActive","location","sensitive","strict","style","to","innerRef"]);return a.createElement(r.s6.Consumer,null,(function(e){e||(0,u.Z)(!1);var n=R||e.location,i=p(s(k,n),n),l=i.pathname,A=l&&l.replace(/([.+*?=^!:${}()[\]|/\\])/g,"\\$1"),K=A?(0,r.LX)(n.pathname,{path:A,exact:h,sensitive:b,strict:O}):null,N=!!(C?C(K,n):K),P="function"==typeof y?y(N):y,_="function"==typeof j?j(N):j;N&&(P=function(){for(var e=arguments.length,t=new Array(e),n=0;n<e;n++)t[n]=arguments[n];return t.filter((function(e){return e})).join(" ")}(P,f),_=c({},_,v));var x=c({"aria-current":N&&o||null,className:P,style:_,to:i},E);return d!==g?x.ref=t||w:x.innerRef=w,a.createElement(m,x)}))}))}}]);